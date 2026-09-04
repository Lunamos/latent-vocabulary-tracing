"""Content table for one zoo readout store (ro_<tag>.pt): which tokens did model B promote / suppress
relative to model A, on the union of both models' top-k support, averaged over (position, layer).
Same delta definition as opd/explore/content_table_final.py (logprob_b - logprob_a on union support),
restricted to the response span of probes of one kind, over a layer band and one readout (J / LL / final).

Also reports a CJK share (fraction of the top-N suppressed / promoted tokens containing CJK characters),
the zh->en codebook-translation statistic for H4.

Usage: python content_table.py TAG [--readout J] [--layers 20-26] [--kind math] [--topn 40] [--min_n 20]
Output: zoo/analysis/content/<tag>_<readout>_<kind>_<layers>.json + .txt
"""
import argparse, json, os, re
import numpy as np, torch
import transformers
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--readout", default="J"); ap.add_argument("--layers", default="20-26")
ap.add_argument("--kind", default="math"); ap.add_argument("--support", default="union", choices=["union","a"]); ap.add_argument("--topn", type=int, default=40); ap.add_argument("--min_n", type=int, default=20)
args = ap.parse_args()
d = torch.load(f"{ZOO}/results/ro_{args.tag}.pt", map_location="cpu", weights_only=False)
import re as _re
_ma = d["model_a"]; _mm = _re.search(r"models--([^/]+)--([^/]+)/snapshots", _ma)
tok = transformers.AutoTokenizer.from_pretrained(f"{_mm.group(1)}/{_mm.group(2)}" if _mm else _ma)
VOCAB = max(len(tok), int(max(int(L["top_a"].max()) for st in d["store"].values() for L in (list(st.get("J", st.get("LL", {})).values()) + ([st["final"]] if "final" in st else [])))) + 1)
if args.readout == "final":
    layers = ["final"]
else:
    lo, hi = map(int, args.layers.split("-")); layers = [l for l in d["layers"] if lo <= l <= hi]
if args.readout != "final" and not d["store"][d["records"][0]["key"]].get(args.readout):
    args.readout = "LL" if args.readout == "J" else args.readout   # stores without a J readout (no lens for that family)
sums, cnts = np.zeros(VOCAB), np.zeros(VOCAB, dtype=np.int64)
n_rec = n_pos = 0
for rec in d["records"]:
    if rec["kind"] != args.kind: continue
    st = d["store"][rec["key"]]
    p_lo, p_hi = rec["prompt_len"], st["n_pos"]
    if p_hi <= p_lo: continue
    n_rec += 1; n_pos += p_hi - p_lo
    off = st.get("store_offset", 0)          # stores written after 09-02 22:00 hold only the response span
    p_lo, p_hi = p_lo - off, p_hi - off
    for l in layers:
        L = st["final"] if l == "final" else st[args.readout][l]
        ta = L["top_a"].long().numpy()[p_lo:p_hi]; tb = L["top_b"].long().numpy()[p_lo:p_hi]
        va = L["val_a"].float().numpy()[p_lo:p_hi]; vb = L["val_b"].float().numpy()[p_lo:p_hi]
        lsa = L["lse_a"].float().numpy()[p_lo:p_hi, None]; lsb = L["lse_b"].float().numpy()[p_lo:p_hi, None]
        if "a_at_b" in L:
            aab = L["a_at_b"].float().numpy()[p_lo:p_hi]; baa = L["b_at_a"].float().numpy()[p_lo:p_hi]
            # tokens in top_a: lp_a = va - lsa, lp_b = baa - lsb
            delta_a = (baa - lsb) - (va - lsa)
            np.add.at(sums, ta.ravel(), delta_a.ravel()); np.add.at(cnts, ta.ravel(), 1)
            # tokens in top_b not in top_a
            k = ta.shape[1]
            in_a = (tb[:, :, None] == ta[:, None, :]).any(-1)
            delta_b = (vb - lsb) - (aab - lsa)
            mask = ~in_a
            if args.support == "union":
                np.add.at(sums, tb[mask], delta_b[mask]); np.add.at(cnts, tb[mask], 1)
        else:  # slim LL store: only tokens in BOTH top-k have both logprobs
            k = ta.shape[1]
            eq = (ta[:, :, None] == tb[:, None, :])  # [pos, k, k]
            ia, ja = np.nonzero(eq.any(-1))
            jb = eq[ia, ja].argmax(-1)
            delta = (vb[ia, jb] - lsb[ia, 0]) - (va[ia, ja] - lsa[ia, 0])
            np.add.at(sums, ta[ia, ja], delta); np.add.at(cnts, ta[ia, ja], 1)
mean = np.where(cnts >= args.min_n, sums / np.maximum(cnts, 1), np.nan)
ok = np.where(~np.isnan(mean))[0]
order = ok[np.argsort(mean[ok])]
CJK = re.compile(r"[一-鿿㐀-䶿]")
def row(t): return {"id": int(t), "tok": tok.decode([int(t)]), "delta": float(mean[t]), "n": int(cnts[t]), "cjk": bool(CJK.search(tok.decode([int(t)])))}
sup = [row(t) for t in order[:args.topn]]; amp = [row(t) for t in order[::-1][:args.topn]]
out = {"tag": args.tag, "model_a": d["model_a"], "model_b": d["model_b"], "readout": args.readout, "layers": [str(l) for l in layers],
       "kind": args.kind, "n_records": n_rec, "n_positions": n_pos, "n_tokens_scored": int(len(ok)),
       "cjk_share_suppressed": float(np.mean([r["cjk"] for r in sup])) if sup else None,
       "cjk_share_promoted": float(np.mean([r["cjk"] for r in amp])) if amp else None,
       "cjk_mean_delta": float(np.nanmean([mean[t] for t in ok if CJK.search(tok.decode([int(t)]))])) if len(ok) else None,
       "suppressed": sup, "promoted": amp}
os.makedirs(f"{ZOO}/analysis/content", exist_ok=True)
base = f"{ZOO}/analysis/content/{args.tag}_{args.readout}_{args.kind}_{args.layers if args.readout != 'final' else 'final'}" + ("_Asupport" if args.support == "a" else "")
json.dump(out, open(base + ".json", "w"), indent=1, ensure_ascii=False)
with open(base + ".txt", "w") as f:
    f.write(f"# {args.tag}: {d['model_a']} -> {d['model_b']} | {args.readout} L{args.layers} | {args.kind} | {n_rec} probes, {n_pos} positions\n")
    f.write(f"# CJK share: suppressed {out['cjk_share_suppressed']:.2f} promoted {out['cjk_share_promoted']:.2f}; mean delta of CJK tokens {out['cjk_mean_delta']:+.3f}\n")
    f.write("SUPPRESSED (delta logprob)          | PROMOTED\n")
    for a, b in zip(sup, amp):
        f.write(f"{a['delta']:+6.2f} {a['n']:5d} {a['tok']!r:22s} | {b['delta']:+6.2f} {b['n']:5d} {b['tok']!r}\n")
print(open(base + ".txt").read()[:2500])
