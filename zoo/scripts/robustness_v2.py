"""Probe-set robustness: compare v1 (probes.jsonl) and v2 (probes_v2.jsonl) readouts for the same pairs.
Prints per-pair work-band JS (math/agent), peak layer, locality ratio (in-domain/wikitext) under both sets, plus rank correlations.
Writes paper/robustness_v2.tex."""
import json, glob, math
import numpy as np
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
S = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(f"{ZOO}/results/ro_*_summary.json")}
def band(s, kind, lo=20, hi=26):
    a = s["agg"].get(kind); ro = "J" if a and a.get("J") else "LL"
    return float(np.mean([a[ro][str(l)]["js"] for l in range(lo, hi + 1)])) if a else float("nan")
def peak(s, kind):
    a = s["agg"][kind]; ro = "J" if a.get("J") else "LL"; return max(range(6, 35), key=lambda l: a[ro][str(l)]["js"])
rows = []
for t in sorted(S):
    if not t.startswith("v2_"): continue
    b = t[3:]
    if b not in S: continue
    kind = "agent" if any(k in b for k in ("swelego", "sera", "terminal", "ota2", "seta", "tmax", "sweagile")) else "math"
    v1, v2 = S[b], S[t]
    rows.append((b, kind, band(v1, kind), band(v2, kind), peak(v1, kind), peak(v2, kind), band(v1, kind) / band(v1, "neutral"), band(v2, kind) / band(v2, "neutral")))
print(f"{'pair':18s} {'kind':6s} {'v1 JS':>7s} {'v2 JS':>7s} {'ratio':>6s} {'peak v1/v2':>10s} {'local v1':>8s} {'local v2':>8s}")
for r in rows: print(f"{r[0]:18s} {r[1]:6s} {r[2]:7.4f} {r[3]:7.4f} {r[3]/r[2]:6.2f} {r[4]:4d}/{r[5]:<5d} {r[6]:8.1f} {r[7]:8.1f}")
x = np.log([r[2] for r in rows]); y = np.log([r[3] for r in rows])
def rank(v): return np.argsort(np.argsort(v))
sp = np.corrcoef(rank(x), rank(y))[0, 1]; pe = np.corrcoef(x, y)[0, 1]
print(f"n={len(rows)}  Spearman(log JS) = {sp:.3f}  Pearson(log JS) = {pe:.3f}  median v2/v1 = {np.median([r[3]/r[2] for r in rows]):.2f}  peaks equal within 2 layers: {np.mean([abs(r[4]-r[5])<=2 for r in rows]):.2f}")
with open("/localscratch/zjin350/Documents/jlen/paper/robustness_v2.tex", "w") as f:
    f.write("\\begin{table}[h]\\centering\\scriptsize\\caption{Probe-set robustness: the same pairs read on a disjoint second probe set (different MATH seed, HumanEval tasks 60--89, other agent trajectories, wikitext offset 2000). Work-band JS on the in-domain probes, peak layer and locality ratio (in-domain / wikitext) under both sets.}\\label{tab:v2}\\begin{tabular}{llrrrrrr}\\toprule pair & domain & JS v1 & JS v2 & peak v1 & peak v2 & locality v1 & locality v2 \\\\\\midrule\n")
    for r in rows: f.write(f"{r[0].replace('_', chr(92)+'_')} & {r[1]} & {r[2]:.3f} & {r[3]:.3f} & {r[4]} & {r[5]} & {r[6]:.1f} & {r[7]:.1f} \\\\\n")
    f.write("\\bottomrule\\end{tabular}\\end{table}\n")
