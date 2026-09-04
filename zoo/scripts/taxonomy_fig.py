"""Summary figure for write_taxonomy.py output: where does the moved probability mass go, by vocabulary category.

Reads zoo/analysis/taxonomy/taxonomy.csv. For a chosen readout/kind, draws
  panel A: stacked bars of the *lost* mass (which categories the descendant drains, share of total loss)
  panel B: stacked bars of the *gained* mass (which categories it fills)
  panel C: net relative change per category (heatmap, %), with the parent's composition as the first row.
Usage: python taxonomy_fig.py [--readout J] [--kind math] [--tags t1,t2,...] [--out fig8_taxonomy]
"""
import argparse, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--readout", default="J"); ap.add_argument("--kind", default="math")
ap.add_argument("--tags", default=""); ap.add_argument("--out", default="fig8_taxonomy"); args = ap.parse_args()
rows = [r for r in csv.DictReader(open(f"{ZOO}/analysis/taxonomy/taxonomy.csv")) if r["readout"] == args.readout and r["kind"] == args.kind]
if args.tags: order = [t for t in args.tags.split(",") if t]; rows = sorted([r for r in rows if r["tag"] in order], key=lambda r: order.index(r["tag"]))
if not rows: raise SystemExit("no rows")
CATS = ["cjk", "other_script", "junk", "junk_id", "format", "punct", "number", "math", "code", "discourse", "function", "english", "latin_piece", "special", "other"]
CATS = [c for c in CATS if f"comp_{c}" in rows[0]]
COL = {"cjk": "#d62728", "other_script": "#ff9896", "junk": "#7f7f7f", "junk_id": "#bcbcbc", "format": "#c5b0d5", "punct": "#9467bd", "number": "#17becf",
       "math": "#1f77b4", "code": "#aec7e8", "discourse": "#2ca02c", "function": "#98df8a", "english": "#ff7f0e", "latin_piece": "#ffbb78", "special": "#e377c2", "other": "#dbdb8d"}
LABEL = {"cjk": "CJK", "other_script": "other script", "junk": "byte junk", "junk_id": "junk identifiers", "format": "markdown/format", "punct": "punct/newline",
         "number": "numbers", "math": "math/LaTeX", "code": "code", "discourse": "reasoning markers", "function": "function words", "english": "English words",
         "latin_piece": "latin pieces", "special": "special tokens", "other": "other"}
tags = [r["tag"] for r in rows]
fig, axs = plt.subplots(1, 3, figsize=(16, 0.42 * len(tags) + 2.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
for ax, key, title in ((axs[0], "sup", "lost probability mass: taken from"), (axs[1], "pro", "gained probability mass: given to")):
    left = np.zeros(len(tags))
    for c in CATS:
        v = np.array([float(r[f"{key}_{c}"]) for r in rows]) * 100
        ax.barh(range(len(tags)), v, left=left, color=COL[c], label=LABEL[c], height=0.7); left += v
    ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% of moved mass"); ax.set_title(title, fontsize=10)
# reference: parent composition (first row's comp; all rows with the same parent share it approximately)
ax = axs[2]
M = np.array([[float(r[f"net_{c}"]) * 100 for c in CATS] for r in rows])
M = np.clip(M, -100, 100)
im = ax.imshow(M, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
ax.set_xticks(range(len(CATS))); ax.set_xticklabels([LABEL[c] for c in CATS], rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=8); ax.set_title("net change of the parent's mass in that category (%)", fontsize=10)
for i in range(len(tags)):
    for j, c in enumerate(CATS):
        comp = float(rows[i][f"comp_{c}"]) * 100
        if comp >= 1: ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=6, color="black" if abs(M[i, j]) < 60 else "white")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"What the write moves, by vocabulary category · {args.readout} lens, work band, {args.kind} probes, parent's top-50 support", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
fig.savefig(f"{ZOO}/figs/{args.out}_{args.readout}_{args.kind}.pdf"); fig.savefig(f"{ZOO}/figs/{args.out}_{args.readout}_{args.kind}.png", dpi=140)
print("wrote", f"{ZOO}/figs/{args.out}_{args.readout}_{args.kind}.png")
# text digest: composition of the parent + the biggest movers
print(f"\nparent composition ({rows[0]['tag']} row): " + ", ".join(f"{LABEL[c]} {100*float(rows[0][f'comp_{c}']):.0f}%" for c in CATS if float(rows[0][f"comp_{c}"]) >= 0.02))
print("\nnet change per category (% of the parent's mass in that category; only categories >=2% of parent mass), cross-category share of the moved mass")
for r in rows:
    nets = sorted([c for c in CATS if float(r[f"comp_{c}"]) >= 0.02], key=lambda c: float(r[f"net_{c}"]))
    print(f"{r['tag']:20s} moved {float(r['mass_per_occ']):.3f}/cell cross {100*float(r.get('cross_share', 0)):3.0f}% | " + ", ".join(f"{LABEL[c]} {100*float(r[f'net_{c}']):+.0f}%" for c in nets))
for r in rows:
    lost = sorted(CATS, key=lambda c: -float(r[f"sup_{c}"]))[:3]; gain = sorted(CATS, key=lambda c: -float(r[f"pro_{c}"]))[:3]
    print(f"{r['tag']:20s} moved {float(r['mass_per_occ']):.3f}/cell | drains " + ", ".join(f"{LABEL[c]} {100*float(r[f'sup_{c}']):.0f}%" for c in lost)
          + " | fills " + ", ".join(f"{LABEL[c]} {100*float(r[f'pro_{c}']):.0f}%" for c in gain))
