"""Does the frozen Base lens still read model B coherently?  For each lens layer and both models, overlap@k between
the J-lens (and logit-lens) top-k at a position and the SAME model's own final-logit top-k at that position.
A drop for B relative to A = the ruler reads B less coherently than it reads the model it was fitted on ("melting").
Also: fraction of J top-10 tokens that are 'junk' (single Hebrew/Arabic letters, whitespace runs, bytes) per model.

Usage: python lens_coherence.py TAG [TAG ...]  -> prints per-band table, writes zoo/analysis/coherence_<tag>.json
"""
import sys, json, os, re
import numpy as np, torch, transformers
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
JUNK = re.compile(r"^\s*$|^[֐-׿؀-ۿ]{1,2}$|^\s?[֐-׿؀-ۿ]{1,2}$|�")
vocab_junk = np.array([bool(JUNK.search(tok.decode([i]))) for i in range(len(tok))] + [False] * (152064 - len(tok)))
BANDS = {"early": range(6, 15), "mid": range(15, 20), "work": range(20, 27), "late": range(27, 34)}
K = 10
os.makedirs(f"{ZOO}/analysis", exist_ok=True)
print(f"{'tag':14s} {'kind':8s} {'band':6s} | J∩final A/B (top{K}) | LL∩final A/B | J junk A/B | LL junk A/B")
for tag in sys.argv[1:]:
    d = torch.load(f"{ZOO}/results/ro_{tag}.pt", map_location="cpu", weights_only=False)
    res = {}
    for kind in ("math", "code", "agent", "neutral"):
        acc = {(ro, l): [] for ro in ("J", "LL") for l in d["layers"]}
        for rec in d["records"]:
            if rec["kind"] != kind: continue
            st = d["store"][rec["key"]]
            fa, fb = st["final"]["top_a"].long().numpy()[:, :K], st["final"]["top_b"].long().numpy()[:, :K]
            if fa.shape[0] == 0: continue
            for ro in ("J", "LL"):
                for l in d["layers"]:
                    L = st[ro][l]
                    ja, jb = L["top_a"].long().numpy()[:, :K], L["top_b"].long().numpy()[:, :K]
                    oa = (ja[:, :, None] == fa[:, None, :]).any(-1).mean()
                    ob = (jb[:, :, None] == fb[:, None, :]).any(-1).mean()
                    acc[(ro, l)].append((oa, ob, vocab_junk[ja].mean(), vocab_junk[jb].mean()))
        res[kind] = {}
        for bname, band in BANDS.items():
            vals = np.array([v for (ro, l), vs in acc.items() if ro == "J" and l in band for v in vs])
            vll = np.array([v for (ro, l), vs in acc.items() if ro == "LL" and l in band for v in vs])
            if len(vals) == 0: continue
            m, mll = vals.mean(0), vll.mean(0)
            res[kind][bname] = {"J_overlap_a": m[0], "J_overlap_b": m[1], "LL_overlap_a": mll[0], "LL_overlap_b": mll[1],
                                "J_junk_a": m[2], "J_junk_b": m[3], "LL_junk_a": mll[2], "LL_junk_b": mll[3]}
            print(f"{tag:14s} {kind:8s} {bname:6s} | {m[0]:.3f}/{m[1]:.3f} | {mll[0]:.3f}/{mll[1]:.3f} | {m[2]:.3f}/{m[3]:.3f} | {mll[2]:.3f}/{mll[3]:.3f}")
        # per-layer J overlap curves
        res[kind]["per_layer"] = {str(l): {"J_a": float(np.mean([v[0] for v in acc[("J", l)]])), "J_b": float(np.mean([v[1] for v in acc[("J", l)]])),
                                           "LL_a": float(np.mean([v[0] for v in acc[("LL", l)]])), "LL_b": float(np.mean([v[1] for v in acc[("LL", l)]]))} for l in d["layers"] if acc[("J", l)]}
    json.dump({"tag": tag, "model_b": d["model_b"], "k": K, "res": {k: {b: {kk: float(vv) for kk, vv in v.items()} if b != "per_layer" else v for b, v in r.items()} for k, r in res.items()}},
              open(f"{ZOO}/analysis/coherence_{tag}.json", "w"), indent=1)
