"""Paper figures from zoo/analysis + zoo/results (matplotlib, PDF+PNG into zoo/figs/).
Palette (validated, fixed order): SFT/distill #2a78d6, RL/DPO increment #eb6834, OPD #1baf7a, zero-RL #eda100, reference/other #898781.
"""
import csv, json, glob, os, re, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"; FIG = f"{ZOO}/figs"; os.makedirs(FIG, exist_ok=True)
C = {"sft": "#2a78d6", "rl": "#eb6834", "opd": "#1baf7a", "zero": "#eda100", "ref": "#898781"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42})
S = {}
for f in glob.glob(f"{ZOO}/results/ro_*_summary.json"):
    s = json.load(open(f)); S[s["tag"]] = s
def band(tag, kind="math", ro="J", lo=20, hi=26):
    a = S[tag]["agg"][kind]; ro = ro if a.get(ro) else "LL"
    return float(np.mean([a[ro][str(l)]["js"] for l in range(lo, hi + 1) if str(l) in a[ro]]))
def prof(tag, kind="math", ro="J"):
    a = S[tag]["agg"][kind]; ro = ro if a.get(ro) else "LL"; L = S[tag]["layers"]
    return L, [a[ro][str(l)]["js"] for l in L]
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close(fig); print("wrote", name)

# ---------- Fig 1: layer profiles (Qwen3-8B J-lens | OLMo-3 own J-lens) ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
qwen = [("klear_sft", "Klear-SFT (SFT)", "sft", "-"), ("casc_sft", "Cascade-SFT (SFT)", "sft", "--"), ("it_swelego", "SWE-Lego vs IT (agent SFT)", "sft", ":"),
        ("arm_opd100", "OPD 100 steps (ours)", "opd", "-"), ("opd_sft2opd", "SFT→OPD (public)", "opd", "--"),
        ("klear_sft2rl", "Klear SFT→RL", "rl", "-"), ("casc_sft2rlhf", "Cascade SFT→RLHF", "rl", "--"), ("tmax_sft2rl8b_s500", "tmax SFT→RL s500", "rl", ":"),
        ("dapo27", "DAPO zero-RL", "zero", "-"), ("rulereasoner", "RuleReasoner zero-RL", "zero", "--")]
olmo = [("olmoJ_think_sft", "Think-SFT (SFT)", "sft", "-"), ("olmoJ_sft_s1k", "Think-SFT step 1k", "sft", ":"), ("olmoJ_sft2dpo", "SFT→DPO", "rl", "-"),
        ("olmoJ_dpo2think", "DPO→RLVR", "rl", "--"), ("olmoJ_rlzero_math", "RL-Zero-Math", "zero", "-"), ("olmoJ_rlzero_code", "RL-Zero-Code", "zero", "--")]
for ax, items, title, span in ((axs[0], qwen, "Qwen3-8B family · frozen Base J-lens", (20, 26)), (axs[1], olmo, "OLMo-3 7B family · own J-lens", (14, 22))):
    ax.axvspan(*span, color="#f0efec", zorder=0)
    for tag, lab, cls, ls in items:
        if tag not in S: continue
        L, y = prof(tag); ax.plot(L, y, ls, color=C[cls], lw=1.6, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_title(title, fontsize=8, color=INK, loc="left"); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(fontsize=5.2, frameon=False, ncol=1, loc="lower right", handlelength=1.6, labelspacing=0.25)
axs[0].set_ylabel("JS(descendant ‖ parent) on math probes")
save(fig, "fig1_profiles")

# ---------- Fig 2: write magnitude by contrast ----------
rows = [  # (label, tag, class, kind)
 ("Base→Cascade-SFT", "casc_sft", "sft", "math"), ("Base→Klear-SFT", "klear_sft", "sft", "math"), ("Base→R1-0528 distill", "r1distill", "sft", "math"),
 ("Base→Qwen3-8B (official)", "it", "sft", "math"), ("Base→ODA-Math SFT", "oda_math", "sft", "math"), ("Base→MegaScience SFT", "megascience", "sft", "math"),
 ("IT→SWE-Lego (agent SFT)", "it_swelego", "sft", "agent"), ("IT→Nemotron-Terminal (SFT)", "it_nemo_terminal", "sft", "agent"), ("IT→SERA (agent SFT)", "it_sera", "sft", "agent"),
 ("Base→OPD 100 steps (ours)", "arm_opd100", "opd", "math"), ("Base→offKD 100 (ours)", "arm_offkd100", "sft", "math"), ("Base→SFT 100 (ours)", "arm_sft100", "sft", "math"),
 ("SFT-3000→OPD-200 (public)", "opd_sft2opd", "opd", "math"),
 ("Cascade SFT→RLHF", "casc_sft2rlhf", "rl", "math"), ("Cascade RLHF→IF-RL", "casc_rlhf2ifrl", "rl", "math"), ("Cascade IF→Math-RL", "casc_ifrl2mathrl", "rl", "math"),
 ("Cascade Math→Code-RL", "casc_mathrl2coderl", "rl", "math"), ("Cascade Code→SWE-RL", "casc_coderl2final", "rl", "agent"), ("Klear SFT→RL (GPPO)", "klear_sft2rl", "rl", "math"),
 ("X-Coder SFT→RL", "xcoder_sft2rl", "rl", "code"), ("OpenThinkerAgent SFT→RLOO", "ota2_sft2rl", "rl", "agent"), ("tmax SFT→DPPO s500", "tmax_sft2rl8b_s500", "rl", "agent"),
 ("SWE-AGILE SFT→RLVR", "sweagile_sft2rl", "rl", "agent"), ("SUTRA distil→RLVR", "sutra_distil2rlvr", "rl", "math"), ("IT→SETA-Env RL (RL only)", "it_seta_rl", "rl", "agent"),
 ("Base→RLVR 100 (ours)", "arm_rlvr100", "zero", "math"), ("Base→DAPO-27 (zero-RL)", "dapo27", "zero", "math"), ("Base→PPO-23 (zero-RL)", "ppo23_store", "zero", "math"), ("Base→RuleReasoner (zero-RL)", "rulereasoner", "zero", "math")]
rows = [r for r in rows if r[1] in S]
fig, ax = plt.subplots(figsize=(4.6, 0.19 * len(rows) + 0.8))
ys = np.arange(len(rows))[::-1]
for y, (lab, tag, cls, kind) in zip(ys, rows):
    v = band(tag, kind); ax.barh(y, v, color=C[cls], height=0.62); ax.text(v * 1.15, y, f"{v:.3f}", va="center", fontsize=6, color=INK2)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5); ax.set_xscale("log"); ax.set_xlim(5e-4, 1.5)
ax.set_xlabel("work-band JS (L20–26, J-lens) on the in-domain probe set"); ax.grid(axis="x", color=GRID, lw=0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["sft"], label="SFT / off-policy distillation"), Patch(color=C["opd"], label="on-policy distillation"), Patch(color=C["rl"], label="RL / DPO stage"), Patch(color=C["zero"], label="RL from Base")], fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
save(fig, "fig2_magnitude")

# ---------- Fig 3: locality heatmap ----------
loc_rows = [("Klear-SFT", "klear_sft"), ("Cascade-SFT", "casc_sft"), ("R1 distill", "r1distill"), ("Qwen3-8B official", "it"), ("MegaScience SFT", "megascience"),
            ("SWE-Lego vs IT", "it_swelego"), ("SERA vs IT", "it_sera"), ("OpenThinkerAgent SFT vs IT", "it_ota2_sft"), ("OPD 100 (ours)", "arm_opd100"),
            ("Cascade SFT→RLHF", "casc_sft2rlhf"), ("Cascade IF→Math-RL", "casc_ifrl2mathrl"), ("Klear SFT→RL", "klear_sft2rl"), ("OpenThinkerAgent SFT→RLOO", "ota2_sft2rl"),
            ("tmax SFT→DPPO s500", "tmax_sft2rl8b_s500"), ("SETA-Env RL vs IT", "it_seta_rl"), ("DAPO-27 zero-RL", "dapo27"), ("RuleReasoner zero-RL", "rulereasoner")]
loc_rows = [r for r in loc_rows if r[1] in S and "agent" in S[r[1]]["agg"]]
kinds = ["math", "code", "agent", "neutral"]
M = np.array([[band(t, k) for k in kinds] for _, t in loc_rows]); R = M / M.max(1, keepdims=True)
fig, ax = plt.subplots(figsize=(3.6, 0.24 * len(loc_rows) + 0.7))
im = ax.imshow(R, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7f9fd", "#2a78d6"]), vmin=0, vmax=1, aspect="auto")
for i in range(len(loc_rows)):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]:.3f}" if M[i, j] >= 0.01 else f"{M[i, j]:.4f}", ha="center", va="center", fontsize=5.5, color=INK if R[i, j] < 0.6 else "white")
ax.set_xticks(range(4)); ax.set_xticklabels(["math", "code", "agent", "wikitext"], fontsize=7); ax.set_yticks(range(len(loc_rows))); ax.set_yticklabels([r[0] for r in loc_rows], fontsize=6.5)
ax.set_title("work-band JS by probe domain (row-normalized shade)", fontsize=8, loc="left", color=INK)
save(fig, "fig3_locality")

# ---------- Fig 4: direction alignment heatmap ----------
fa = f"{ZOO}/analysis/alignment_fig_J_math_work.csv"
if os.path.exists(fa):
    rows_ = list(csv.reader(open(fa))); tags = rows_[0][1:]; A = np.array([[float(x) for x in r[1:]] for r in rows_[1:]])
    names = {"arm_sft100": "SFT (ours)", "arm_offkd100": "offKD (ours)", "arm_opd100": "OPD (ours)", "it": "Qwen3-8B", "r1distill": "R1 distill", "klear_sft": "Klear-SFT", "casc_sft": "Cascade-SFT",
             "oda_math": "ODA-Math", "swelego": "SWE-Lego", "sera": "SERA", "nemo_terminal": "Nemotron-Term.", "ota2_sft": "OT-Agent SFT", "arm_rlvr100": "RLVR (ours)", "dapo27": "DAPO", "ppo23_store": "PPO", "rulereasoner": "RuleReasoner"}
    cls = {t: ("zero" if t in ("arm_rlvr100", "dapo27", "ppo23_store", "rulereasoner") else "opd" if t == "arm_opd100" else "sft") for t in tags}
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#eb6834", "#f0efec", "#2a78d6"])
    ax.imshow(A, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(tags)):
        for j in range(len(tags)):
            if i != j: ax.text(j, i, f"{A[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=4.6, color=INK)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([names.get(t, t) for t in tags], rotation=90, fontsize=6); ax.set_yticks(range(len(tags))); ax.set_yticklabels([names.get(t, t) for t in tags], fontsize=6)
    for lab, t in zip(ax.get_yticklabels(), tags): lab.set_color(C[cls[t]])
    for lab, t in zip(ax.get_xticklabels(), tags): lab.set_color(C[cls[t]])
    ax.set_title("edit-direction alignment on Base's top-50 support (J, L20–26, math)", fontsize=7.5, loc="left", color=INK)
    save(fig, "fig4_alignment")

# ---------- Fig 5: trajectories ----------
def steps(pattern, kind="math", lo=20, hi=26, ro="J"):
    out = []
    for t in S:
        m = re.fullmatch(pattern, t)
        if m: out.append((int(m.group(1)), band(t, kind, ro, lo, hi)))
    return sorted(out)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
ax = axs[0]
for pat, lab, cls, ls in ((r"dapo_s(\d+)", "DAPO (math, work band)", "zero", "-"), (r"ppo_s(\d+)", "PPO (math, work band)", "zero", "--")):
    d = [(x, y) for x, y in steps(pat) if y > 0]; ax.plot([x for x, _ in d], [y for _, y in d], ls, marker="o", ms=3, color=C[cls], label=lab)
ax.set_title("zero-RL from Qwen3-8B-Base", fontsize=8, loc="left", color=INK); ax.set_xlabel("released checkpoint index"); ax.set_ylabel("work-band JS vs parent")
ax = axs[1]
d = steps(r"q35_it2tmax_s(\d+)", "agent", 24, 30, "LL"); ax.plot([x for x, _ in d], [y for _, y in d], "-", marker="o", ms=3, color=C["rl"], label="tmax-9b DPPO (agent, late band)")
d = steps(r"tmax_sft2rl8b_s(\d+)", "agent", 20, 26); ax.plot([x for x, _ in d], [y for _, y in d], "--", marker="o", ms=3, color=C["rl"], label="tmax-8b DPPO after SFT (agent, work band)")
ax.set_title("RL-only agent training", fontsize=8, loc="left", color=INK); ax.set_xlabel("RL step")
ax = axs[2]
sft = [(1000, band("olmoJ_sft_s1k")), (5000, band("olmoJ_sft_s5k")), (43000, band("olmoJ_think_sft"))] if "olmoJ_sft_s5k" in S else []
sftn = [(1000, band("olmoJ_sft_s1k", "neutral")), (5000, band("olmoJ_sft_s5k", "neutral")), (43000, band("olmoJ_think_sft", "neutral"))] if sft else []
if sft:
    ax.plot([x for x, _ in sft], [y for _, y in sft], "-", marker="o", ms=3, color=C["sft"], label="OLMo Think-SFT, math")
    ax.plot([x for x, _ in sftn], [y for _, y in sftn], ":", marker="o", ms=3, color=C["sft"], label="OLMo Think-SFT, wikitext")
rz = [(300, band("olmo_rlzero_math_s300", "math", "LL", 14, 20)), (1000, band("olmo_rlzero_math_s1000", "math", "LL", 14, 20)), (1900, band("olmo_rlzero_math", "math", "LL", 14, 20))]
ax.plot([x for x, _ in rz], [y for _, y in rz], "-", marker="o", ms=3, color=C["zero"], label="OLMo RL-Zero-Math, math (LL)")
ax.set_xscale("log"); ax.set_title("OLMo-3: SFT steps vs RL-Zero steps", fontsize=8, loc="left", color=INK); ax.set_xlabel("training step")
for ax in axs:
    ax.set_yscale("log"); ax.grid(axis="y", color=GRID, lw=0.5); ax.legend(fontsize=5.5, frameon=False)
save(fig, "fig5_trajectories")

# ---------- Fig 6: write vs behavior ----------
ev = {}
for f in glob.glob(f"{ZOO}/results/eval_*_32k.json"):
    e = json.load(open(f)); ev[e["tag"].replace("_32k", "")] = e["aime_avg"]
pts = [  # (label, x-tag for write, parent eval tag, child eval tag, class)
 ("Klear SFT→RL", "klear_sft2rl", "klear_sft", "klear_rl", "rl"), ("SFT-3000→OPD-200", "opd_sft2opd", "opd_sft3000", "opd_step200", "opd"),
 ("Base→DAPO-27", "dapo27", "base", "dapo27", "zero"), ("Base→DAPO-13", "dapo13", "base", "dapo13", "zero"),
 ("Base→Klear-SFT", "klear_sft", "base", "klear_sft", "sft"), ("Base→Cascade", "casc_final", "base", "casc_final", "sft"), ("Base→R1 distill", "r1distill", "base", "r1distill", "sft"),
 ("Base→Qwen3-8B", "it", "base", "it", "sft"), ("Base→SFT-3000", "opd_sft3000", "base", "opd_sft3000", "sft")]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for lab, wt, pa, ch, cls in pts:
    if wt not in S or pa not in ev or ch not in ev: continue
    x = band(wt); y = (ev[ch] - ev[pa]) * 100
    ax.scatter(x, y, color=C[cls], s=28, zorder=3); ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.5, color=INK2)
ax.set_xscale("log"); ax.set_xlabel("work-band JS written by the stage"); ax.set_ylabel("AIME24+25 avg@4 gain (points, 30k budget)")
ax.grid(color=GRID, lw=0.5); ax.set_title("capability bought per unit of rewriting", fontsize=8, loc="left", color=INK)
save(fig, "fig6_write_vs_gain")
