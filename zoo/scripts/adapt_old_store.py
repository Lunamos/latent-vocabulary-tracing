"""Convert a July difflens store (opd/results/difflens_<old>.pt, optional difffinal_<old2>_final.pt) into the zoo ro_<tag> format so
sharpening.py / edit_alignment.py / rl_direction.py / content_table.py / analyze_summaries.py run unchanged on our controlled arms.
Old and new math/neutral probes are the same texts (same seed/protocol), so positions align after slicing to the response span.
Usage: python adapt_old_store.py OLD_DIFFLENS_TAG NEW_TAG [OLD_DIFFFINAL_TAG]
  e.g. python adapt_old_store.py b0_vs_opd100_baselens arm_opd100 b0_vs_opd100_final
"""
import sys, os, json, torch
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"; OPD = "/localscratch/zjin350/Documents/jlen/opd/results"
old, new = sys.argv[1], sys.argv[2]; fin = sys.argv[3] if len(sys.argv) > 3 else None
d = torch.load(f"{OPD}/difflens_{old}.pt", map_location="cpu", weights_only=False)
df = torch.load(f"{OPD}/difffinal_{fin}.pt", map_location="cpu", weights_only=False) if fin else None
layers = [l for l in d["layers"] if 6 <= l <= 34]
records, store = [], {}
for rec in d["records"]:
    st = d["store"][rec["key"]]; p0, n = rec["prompt_len"], st["n_pos"]
    span = slice(p0, n)
    ns = {"n_pos": n, "input_ids": st["input_ids"], "store_offset": p0, "J": {}, "LL": {}}
    for l in layers:
        L = st["layers"][l]; ns["J"][l] = {k: v[span] for k, v in L.items()}
    if df is not None:
        F = df["store"][rec["key"]]["layers"]["final"]; ns["final"] = {k: v[span] for k, v in F.items()}
    store[rec["key"]] = ns
    pl = {"J": {str(l): {k: rec["per_layer"][l][k] for k in ("kl_ab", "kl_ba", "js", "jaccard")} | {
        "js_resp": float(st["layers"][l]["js"][span].float().mean()) if n > p0 else None,
        "kl_ab_resp": float(st["layers"][l]["kl_ab"][span].float().mean()) if n > p0 else None} for l in layers}, "LL": {}}
    r = {"key": rec["key"], "kind": rec["kind"], "prompt_len": p0, "meta": rec["meta"], "n_pos": n, "per_layer": pl, "hidden": {},
         "faith": {"J": {}, "LL": {}}, "nll": {"a": None, "b": None},
         "final": {"js": float(df["store"][rec["key"]]["layers"]["final"]["js"].float().mean()) if df is not None else float("nan"),
                   "kl_ab": float(df["store"][rec["key"]]["layers"]["final"]["kl_ab"].float().mean()) if df is not None else float("nan"),
                   "kl_ba": float("nan"), "jaccard": float("nan"),
                   "js_resp": float(df["store"][rec["key"]]["layers"]["final"]["js"][span].float().mean()) if df is not None and n > p0 else None,
                   "kl_ab_resp": None}}
    records.append(r)
agg = {}
for kind in ("math", "neutral"):
    rs = [r for r in records if r["kind"] == kind]
    a = {"n": len(rs), "J": {}, "LL": {}, "hidden": {}, "faith": {"J": {}, "LL": {}}}
    for l in layers:
        a["J"][str(l)] = {k: sum(r["per_layer"]["J"][str(l)][k] for r in rs) / len(rs) for k in ("kl_ab", "kl_ba", "js", "jaccard")}
        v = [r["per_layer"]["J"][str(l)]["js_resp"] for r in rs if r["per_layer"]["J"][str(l)]["js_resp"] is not None]
        a["J"][str(l)]["js_resp"] = sum(v) / len(v) if v else None
    a["final"] = {k: sum(r["final"][k] for r in rs) / len(rs) for k in ("js", "kl_ab")}
    a["nll"] = {"a": None, "b": None}; agg[kind] = a
os.makedirs(f"{ZOO}/results", exist_ok=True)
torch.save({"records": records, "store": store, "layers": layers, "model_a": d["model_a"], "model_b": "july-arm:" + d["model_b"], "lens": d.get("lens_path")},
           f"{ZOO}/results/ro_{new}.pt")
json.dump({"model_a": d["model_a"], "model_b": "july-arm:" + d["model_b"], "tag": new, "lens": d.get("lens_path"), "layers": layers, "readouts": ["J"],
           "source": f"difflens_{old}.pt" + (f" + difffinal_{fin}.pt" if fin else ""), "agg": agg, "records": records}, open(f"{ZOO}/results/ro_{new}_summary.json", "w"), indent=1)
wb = sum(agg["math"]["J"][str(l)]["js"] for l in range(20, 27)) / 7
print(f"{new}: math J work-band JS {wb:.4f}, neutral {sum(agg['neutral']['J'][str(l)]['js'] for l in range(20,27))/7:.4f}, final math JS {agg['math']['final']['js']:.4f}")
