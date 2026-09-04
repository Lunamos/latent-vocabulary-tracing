"""Edit-direction alignment between zoo readouts that share the same reference model A (Base) and probes.

For pair stores ro_<t1>.pt and ro_<t2>.pt: at every (probe, layer, position) in the stored response span,
d_i = logprob_Bi - logprob_A on A's top-50 support (identical support for both since A is the same model):
  d = (b_at_a - lse_b) - (val_a - lse_a)
alignment = weighted mean over positions of Pearson(d_1, d_2), weight = sqrt(|d_1| * |d_2|) (as opd/explore/edit_alignment_v2.py).
Reported per probe kind and per band (early 6-14 / mid 15-19 / work 20-26 / late 27-33) for readout J (and 'final').

Usage: python edit_alignment.py TAG1 TAG2 [TAG3 ...]  -> zoo/analysis/alignment/<readout>_<kind>_<band>.csv (matrix over tags)
Set ALIGNMENT_SUBDIR to keep a named analysis run in its own subdirectory.
"""
import sys, os, json, itertools
import numpy as np, torch
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
tags = sys.argv[1:]
BANDS = {"early": range(6, 15), "mid": range(15, 20), "work": range(20, 27), "late": range(27, 34)}
KINDS = ("math", "code", "agent", "neutral")


def deltas(tag):
    d = torch.load(f"{ZOO}/results/ro_{tag}.pt", map_location="cpu", weights_only=False)
    out = {}  # (key, readout, layer) -> [pos, 50] delta on A support
    for rec in d["records"]:
        st = d["store"][rec["key"]]
        for ro in ("J", "final"):
            if ro == "final" and "final" not in st: continue
            Ls = [("final", st["final"])] if ro == "final" else [(l, st["J"][l]) for l in d["layers"]]
            for l, L in Ls:
                va, lsa = L["val_a"].float(), L["lse_a"].float()[:, None]
                baa, lsb = L["b_at_a"].float(), L["lse_b"].float()[:, None]
                out[(rec["key"], ro, l)] = ((baa - lsb) - (va - lsa)).numpy()
    kinds = {rec["key"]: rec["kind"] for rec in d["records"]}
    return out, kinds, d["model_a"], d["layers"]


D = {t: deltas(t) for t in tags}
import re as _re
def canon(m):
    mm = _re.search(r"models--([^/]+)--([^/]+)/snapshots", m)
    return f"{mm.group(1)}/{mm.group(2)}" if mm else m
ma = {canon(D[t][2]) for t in tags}
assert len(ma) == 1, f"different reference models: {ma}"
layers = D[tags[0]][3]; kinds = D[tags[0]][1]


def align(t1, t2, ro, kind, band):
    num = den = 0.0
    for key, k in kinds.items():
        if k != kind: continue
        Ls = ["final"] if ro == "final" else [l for l in layers if l in band]
        for l in Ls:
            a = D[t1][0].get((key, ro, l)); b = D[t2][0].get((key, ro, l))
            if a is None or b is None or a.shape != b.shape or a.shape[0] == 0: continue
            ac = a - a.mean(1, keepdims=True); bc = b - b.mean(1, keepdims=True)
            r = (ac * bc).sum(1) / (np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
            w = np.sqrt(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            num += float((w * r).sum()); den += float(w.sum())
    return num / den if den > 0 else float("nan")


alignment_root = f"{ZOO}/analysis/alignment"
alignment_subdir = os.environ.get("ALIGNMENT_SUBDIR", "").strip().strip("/")
if alignment_subdir:
    alignment_root = os.path.join(alignment_root, alignment_subdir)
os.makedirs(alignment_root, exist_ok=True)
for ro in ("J", "final"):
    for kind in KINDS:
        for bname, band in (BANDS.items() if ro == "J" else [("final", None)]):
            M = np.full((len(tags), len(tags)), np.nan)
            for i, j in itertools.combinations_with_replacement(range(len(tags)), 2):
                M[i, j] = M[j, i] = align(tags[i], tags[j], ro, kind, band)
            path = f"{alignment_root}/{ro}_{kind}_{bname}.csv"
            with open(path, "w") as f:
                f.write("tag," + ",".join(tags) + "\n")
                for i, t in enumerate(tags):
                    f.write(t + "," + ",".join(f"{x:.3f}" for x in M[i]) + "\n")
            if (ro == "J" and bname == "work") or ro == "final":
                print(f"== {ro} {kind} {bname}"); print(open(path).read())
