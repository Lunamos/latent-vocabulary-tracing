"""Aggregate zoo/results/ro_*_summary.json into tables + figures (pure CPU).

Outputs (zoo/analysis/):
  bands.csv          one row per (tag, kind, readout): mean JS in bands early L6-14 / mid L15-19 / work L20-26 /
                     late L27-33, final JS, peak layer, hidden cos@22, faithfulness (KL final||J at L22, a vs b), nll a/b
  profiles_<kind>.png   per-layer J-lens JS profiles for all Base-referenced pairs
  locality.csv       work-band J JS by probe kind (math/code/agent/neutral) per tag + ratios to neutral
  faith.csv          per-layer KL(final||J) for model b vs model a (does the ruler melt?)
Usage: python analyze_summaries.py [--pattern 'ro_*_summary.json']
"""
import argparse, glob, json, os, csv
import numpy as np
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
ap = argparse.ArgumentParser(); ap.add_argument("--pattern", default="ro_*_summary.json"); args = ap.parse_args()
OUT = f"{ZOO}/analysis"; os.makedirs(OUT, exist_ok=True)
BANDS36 = {"early": range(6, 15), "mid": range(15, 20), "work": range(20, 27), "late": range(27, 34)}
BANDS28 = {"early": range(5, 12), "mid": range(12, 15), "work": range(15, 21), "late": range(21, 27)}  # Qwen3-1.7B (28 layers): same depth fractions as the 36-layer bands


def bands_for(layers):
    return BANDS28 if max(layers) <= 26 else BANDS36

rows, loc, faith = [], [], []
profiles = {}
for f in sorted(glob.glob(f"{ZOO}/results/{args.pattern}")):
    s = json.load(open(f)); tag = s["tag"]; layers = s["layers"]; BANDS = bands_for(layers)
    for kind, a in s["agg"].items():
        for ro in ("J", "LL"):
            if ro not in a or not a[ro]: continue
            js = {int(l): a[ro][str(l)]["js"] for l in layers}
            jsr = {int(l): a[ro][str(l)].get("js_resp") for l in layers if a[ro][str(l)].get("js_resp") is not None}
            band = {b: float(np.mean([js[l] for l in r if l in js])) for b, r in BANDS.items()}
            peak = max(js, key=js.get)
            rows.append({"tag": tag, "model_a": s["model_a"].split("/")[-1], "model_b": s["model_b"].split("/")[-1],
                         "kind": kind, "readout": ro, **{f"js_{b}": v for b, v in band.items()},
                         "js_work_resp": float(np.mean([jsr[l] for l in BANDS["work"] if l in jsr])) if jsr else None,
                         "peak_layer": peak, "js_peak": js[peak], "js_final": a["final"]["js"],
                         "js_final_resp": a["final"].get("js_resp"), "kl_final_ab": a["final"].get("kl_ab"),
                         "cos22": a.get("hidden", {}).get("22", {}).get("cos"), "cka22": a.get("hidden", {}).get("22", {}).get("cka"),
                         "dnorm22": a.get("hidden", {}).get("22", {}).get("dnorm_rel"),
                         "faith22_a": a.get("faith", {}).get(ro, {}).get("22", {}).get("a"), "faith22_b": a.get("faith", {}).get(ro, {}).get("22", {}).get("b"),
                         "nll_a": a.get("nll", {}).get("a"), "nll_b": a.get("nll", {}).get("b"), "n": a["n"]})
            if ro == "J":
                profiles.setdefault(kind, {})[tag] = (s["model_a"], [js[l] for l in layers])
        ro0 = "J" if a.get("J") else "LL"
        wb = float(np.mean([a[ro0][str(l)]["js"] for l in BANDS["work"] if str(l) in a[ro0]]))
        loc.append({"tag": tag, "kind": kind, "js_work_J": wb, "js_final": a["final"]["js"]})
        for l in layers:
            fj = a.get("faith", {}).get("J", {}).get(str(l), {}); fl = a.get("faith", {}).get("LL", {}).get(str(l), {})
            faith.append({"tag": tag, "kind": kind, "layer": l, "J_a": fj.get("a"), "J_b": fj.get("b"), "LL_a": fl.get("a"), "LL_b": fl.get("b")})


def write_csv(path, rs):
    if not rs: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)


write_csv(f"{OUT}/bands.csv", rows)
write_csv(f"{OUT}/faith.csv", faith)
# locality matrix: tag x kind (work-band J JS), plus ratio in-domain/neutral
tags = sorted(set(r["tag"] for r in loc)); kinds = ["math", "code", "agent", "neutral"]
M = {t: {k: None for k in kinds} for t in tags}
for r in loc: M[r["tag"]][r["kind"]] = r["js_work_J"]
lrows = []
for t in tags:
    d = {"tag": t, **{k: M[t][k] for k in kinds}}
    n = M[t]["neutral"]
    for k in ("math", "code", "agent"):
        d[f"{k}/neutral"] = (M[t][k] / n) if (n and M[t][k] is not None) else None
    lrows.append(d)
write_csv(f"{OUT}/locality.csv", lrows)

# figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    layers = json.load(open(sorted(glob.glob(f"{ZOO}/results/{args.pattern}"))[0]))["layers"]
    for kind, d in profiles.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for tag, (ma, prof) in sorted(d.items()):
            if not ma.endswith("Qwen3-8B-Base"): continue
            ax.plot(layers, prof, marker=".", label=tag)
        ax.axvspan(20, 26, color="0.9"); ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_ylabel("J-lens JS vs Base")
        ax.set_title(f"{kind} probes · frozen Base lens"); ax.legend(fontsize=6, ncol=3); fig.tight_layout()
        fig.savefig(f"{OUT}/profiles_{kind}.png", dpi=130); plt.close(fig)
except Exception as e:
    print("figure skipped:", e)

# console digest
print(f"{'tag':18s} {'kind':8s} {'J work':>8s} {'J late':>8s} {'J early':>8s} {'final':>8s} {'peak':>5s} {'cos22':>6s} {'faith22 a/b':>14s} {'nll a/b':>12s}")
for r in rows:
    if r["readout"] != "J": continue
    f = lambda v, w=6: (f"{v:{w}.3f}" if isinstance(v, (int, float)) and v is not None else "-".rjust(w))
    print(f"{r['tag']:18s} {r['kind']:8s} {r['js_work']:8.4f} {r['js_late']:8.4f} {r['js_early']:8.4f} {r['js_final']:8.4f} {r['peak_layer']:5d} {f(r['cos22'])} {f(r['faith22_a'])}/{f(r['faith22_b'])} {f(r['nll_a'],5)}/{f(r['nll_b'],5)}")
print(f"wrote {OUT}/bands.csv locality.csv faith.csv (+profiles_*.png)")
