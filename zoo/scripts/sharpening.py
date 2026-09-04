"""I1 test: is model B's write on A's top-50 support a SHARPENING of A's own distribution?

At every stored (probe, layer, position): d = logp_B - logp_A on A's top-50 (A = reference, B = descendant),
s = logp_A - mean(logp_A)  (the direction that lowers A's temperature; sharpening = positive correlation).
Reported per probe kind, per band, readout J and 'final':
  r_sharp   = weighted mean Pearson(d, s)                      (weight = |d|)
  frac_var  = fraction of ||d||^2 explained by its projection on s (mean over positions)
  r_resid   = for pairs of tags: alignment of the residuals d - proj_s(d)  (is what's left data-specific?)
Usage: python sharpening.py TAG [TAG ...]  -> prints table, writes zoo/analysis/sharpening.csv (append)
"""
import sys, os, csv, itertools
import numpy as np, torch
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
BANDS = {"early": range(6, 15), "mid": range(15, 20), "work": range(20, 27), "late": range(27, 34), "final": ["final"]}
KINDS = ("math", "code", "agent", "neutral")


def load(tag):
    d = torch.load(f"{ZOO}/results/ro_{tag}.pt", map_location="cpu", weights_only=False)
    out = {}
    for rec in d["records"]:
        st = d["store"][rec["key"]]
        for l in list(d["layers"]) + (["final"] if "final" in st else []):
            L = st["final"] if l == "final" else st["J"][l]
            va, lsa = L["val_a"].float().numpy(), L["lse_a"].float().numpy()[:, None]
            baa, lsb = L["b_at_a"].float().numpy(), L["lse_b"].float().numpy()[:, None]
            lpa = va - lsa
            dlt = (baa - lsb) - lpa
            out[(rec["key"], l)] = (dlt, lpa - lpa.mean(1, keepdims=True), rec["kind"])
    return out, d["model_b"].split("/")[-1]


def stats(D, kind, band):
    rs, ws, fv, resid = [], [], [], {}
    for (key, l), (dlt, s, k) in D.items():
        if k != kind or l not in band: continue
        dc = dlt - dlt.mean(1, keepdims=True)
        nd, ns = np.linalg.norm(dc, axis=1), np.linalg.norm(s, axis=1)
        r = (dc * s).sum(1) / (nd * ns + 1e-9)
        w = np.linalg.norm(dlt, axis=1)
        rs.append(r * w); ws.append(w)
        proj = ((dc * s).sum(1, keepdims=True) / (ns[:, None] ** 2 + 1e-9)) * s
        fv.append((np.linalg.norm(proj, axis=1) ** 2) / (nd ** 2 + 1e-9) * w)
        resid[(key, l)] = dc - proj
    if not ws: return None
    W = np.concatenate(ws); return float(np.concatenate(rs).sum() / W.sum()), float(np.concatenate(fv).sum() / W.sum()), resid


tags = sys.argv[1:]
data = {t: load(t) for t in tags}
os.makedirs(f"{ZOO}/analysis", exist_ok=True)
rows = []
print(f"{'tag':16s} {'kind':8s} " + " ".join(f"{b:>12s}" for b in BANDS) + "   (r_sharp / frac_var)")
RES = {}
for t in tags:
    D, name = data[t]
    for kind in KINDS:
        cells = []
        for b, band in BANDS.items():
            st = stats(D, kind, band)
            if st is None: cells.append("      -     "); continue
            r, f, resid = st; RES[(t, kind, b)] = resid
            cells.append(f"{r:+.2f}/{f:.2f}   ")
            rows.append({"tag": t, "model_b": name, "kind": kind, "band": b, "r_sharp": r, "frac_var": f})
        print(f"{t:16s} {kind:8s} " + " ".join(cells))
with open(f"{ZOO}/analysis/sharpening.csv", "a") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()));
    if f.tell() == 0: w.writeheader()
    w.writerows(rows)
# residual alignment between tags (work band, math)
if len(tags) > 1:
    print("\nresidual alignment after removing the sharpening component (J work band, math): raw / residual")
    for t1, t2 in itertools.combinations(tags, 2):
        R1, R2 = RES.get((t1, "math", "work")), RES.get((t2, "math", "work"))
        if not R1 or not R2: continue
        num = den = num0 = den0 = 0.0
        for k in R1:
            if k not in R2: continue
            a, b = R1[k], R2[k]; ra, rb = data[t1][0][k][0], data[t2][0][k][0]
            ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
            w = np.sqrt(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1))
            r = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            r0 = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-9)
            num += (w * r).sum(); den += w.sum(); num0 += (w * r0).sum(); den0 += w.sum()
        if den: print(f"  {t1:14s} ~ {t2:14s}: {num0/den0:+.3f} / {num/den:+.3f}")
