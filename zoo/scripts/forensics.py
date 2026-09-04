"""I2: recipe forensics from scale-free write signatures (leave-one-out).
Classes: 'sft' (SFT / distillation write, vs Base or vs IT), 'rl_inc' (an RL/DPO stage read directly from its SFT/IT parent),
'zero_rl' (RL directly from Base). Features (all scale-free): peak layer, late/work, early/work, mid/work, final/work,
neutral/math, code/math, agent/math band ratios (J work band), nll_b - nll_a on math and neutral. Magnitude (work JS) is reported
separately so we can show classification with and without it.
Usage: python forensics.py  -> prints LOO accuracy and confusions; writes zoo/analysis/forensics.csv
"""
import csv, math, json, numpy as np
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
LABEL = {}
for t in "it r1distill klear_sft casc_sft xcoder_sft oda_math megascience blossom huatuo klear_think drkernel_sft sutra_distil ota_sft sera webexplayer webexplorer orchestrator it_ota_sft it_sera it_webexplorer arm_sft100 arm_offkd100 arm_opd100 arm_sft25 arm_offkd25 arm_opd25 opd_sft3000 swelego it_swelego sera_ga it_sera_ga nemo_terminal it_nemo_terminal terminal_lego it_terminal_lego agentforge_sft it_agentforge_sft ota2_sft it_ota2_sft sweagile_sft it_sweagile_sft tmax_sft8b it_tmax_sft8b".split(): LABEL[t] = "sft"
for t in "klear_sft2rl xcoder_sft2rl casc_sft2rlhf casc_rlhf2ifrl casc_ifrl2mathrl casc_mathrl2coderl casc_coderl2final stepfun_rlvr2pacore ota_sft2rl it_orchestrator sutra_distil2rlvr drkernel_sft2rl agentforge_sft2rl ota2_sft2rl sweagile_sft2rl tmax_sft2rl8b it_seta_rl opd_sft2opd".split(): LABEL[t] = "rl_inc"
for t in "dapo13 dapo27 rulereasoner arm_rlvr100 arm_rlvr500".split(): LABEL[t] = "zero_rl"
rows = {}
for r in csv.DictReader(open(f"{ZOO}/analysis/bands.csv")):
    if r["readout"] != "J": continue
    rows.setdefault(r["tag"], {})[r["kind"]] = r
feats, names, tags, mags = [], None, [], []
for tag, kinds in rows.items():
    if tag not in LABEL or "math" not in kinds or "neutral" not in kinds: continue
    m, n = kinds["math"], kinds["neutral"]
    f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else float("nan")
    w = f(m, "js_work")
    x = {"peak": f(m, "peak_layer"), "late/work": math.log(f(m, "js_late") / w), "early/work": math.log(f(m, "js_early") / w),
         "mid/work": math.log(f(m, "js_mid") / w), "final/work": math.log(f(m, "js_final") / w),
         "neutral/math": math.log(f(n, "js_work") / w),
         "code/math": math.log(f(kinds["code"], "js_work") / w) if "code" in kinds else 0.0,
         "agent/math": math.log(f(kinds["agent"], "js_work") / w) if "agent" in kinds else 0.0,
         "dnll_math": (f(m, "nll_b") - f(m, "nll_a")) if m["nll_b"] not in ("", "None") else 0.0,
         "dnll_neutral": (f(n, "nll_b") - f(n, "nll_a")) if n["nll_b"] not in ("", "None") else 0.0}
    names = list(x.keys())
    if not all(math.isfinite(v) for v in x.values()) or not math.isfinite(math.log(w)): continue
    feats.append([x[k] for k in names]); tags.append(tag); mags.append(math.log(w))
X = np.array(feats); y = np.array([LABEL[t] for t in tags]); classes = sorted(set(y))
def loo(Xm):
    Z = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9); pred = []
    for i in range(len(Z)):
        tr = np.arange(len(Z)) != i
        cents = {c: Z[tr][y[tr] == c].mean(0) for c in classes if (y[tr] == c).any()}
        pred.append(min(cents, key=lambda c: np.linalg.norm(Z[i] - cents[c])))
    return np.array(pred)
for name, Xm in (("scale-free features only", X), ("+ log magnitude", np.column_stack([X, mags]))):
    p = loo(Xm); acc = (p == y).mean()
    print(f"{name}: LOO nearest-centroid accuracy {acc:.2f} on n={len(y)} ({dict(zip(*np.unique(y, return_counts=True)))})")
    for c in classes:
        idx = y == c; print(f"   true {c:8s}: " + ", ".join(f"{pc}={int((p[idx]==pc).sum())}" for pc in classes))
    wrong = [(t, yt, pt) for t, yt, pt in zip(tags, y, p) if yt != pt]; print("   misclassified:", wrong)
with open(f"{ZOO}/analysis/forensics.csv", "w") as fcsv:
    w = csv.writer(fcsv); w.writerow(["tag", "label", "log_work_js"] + names)
    for t, l, mg, fv in zip(tags, y, mags, feats): w.writerow([t, l, f"{mg:.3f}"] + [f"{v:.3f}" for v in fv])
# per-class feature means (for the paper table)
Z = X
print("\nclass means of scale-free features:"); print("         " + " ".join(f"{n:>12s}" for n in names))
for c in classes: print(f"{c:8s} " + " ".join(f"{v:12.2f}" for v in Z[y == c].mean(0)))
