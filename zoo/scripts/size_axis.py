"""Size axis: compare what the same recipe writes at 1.7B / 4B / 8B, on depth-normalised layer profiles.

Reads zoo/results/ro_<tag>_summary.json. Layer index -> depth fraction (layer+1)/n_layers so that the 36-layer
work band 20-26 and the 28-layer band 15-20 both map to ~0.58-0.75.
Outputs: zoo/analysis/size_axis.csv (one row per tag x kind: work-band J/LL JS, final JS, peak depth, locality)
         zoo/figs/fig7_size_axis.pdf/png (math-probe J profiles vs depth, one panel per size, colour = recipe)
Usage: python size_axis.py
"""
import json, os, glob, csv
import numpy as np
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ZOO}/analysis"; FIG = f"{ZOO}/figs"; os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# tag -> (size, recipe, label). recipe in {it, sft, opd, zero_rl, rl_inc, dpo, other}
T = {}
def add(size, recipe, **kw):
    for tag, label in kw.items(): T[tag] = (size, recipe, label)
add("1.7B", "it", q17_it="Qwen3-1.7B (official)")
add("1.7B", "sft", q17_sft_ot3="SFT OpenThoughts3", q17_ultrachat="SFT UltraChat", q17_capsd_cap2k="SFT math 2k", q17_capsd_cap4k="SFT math 4k",
    q17_capsd_ppl2k="SFT math 2k (ppl)", q17_capsd_ppl4k="SFT math 4k (ppl)", q17_sft_lora400k="SFT LoRA 400k", q17_sft_lora114k="SFT LoRA 114k", q17_iol="SFT IOL")
add("1.7B", "opd", q17_opd_dapo="OPD DAPO-math (thunlp)", q17_opd_deepmath="OPD DeepMath", q17_opd_countdown_s20="OPD countdown s20",
    q17_opd_countdown_s100="OPD countdown s100", q17_opd_countdown_s150="OPD countdown s150", q17_opd_kk_s150="OPD KK s150", q17_opd_zebra_s150="OPD zebra s150", q17_opsa="OPSA")
add("1.7B", "zero_rl", q17_base2frpo_dapo="FRPO DAPO-noKL s200", q17_grpo_best="GRPO s80", q17_grpo_end="GRPO s136", q17_ttrl_best="TTRL (no labels)")
add("1.7B", "rl_inc", q17_it2mbpp_grpo="IT->GRPO mbpp", q17_it2wordle_grpo="IT->GRPO wordle")
add("1.7B", "sft_on_it", q17_it2dualmind="IT->agentic distill", q17_it2kimina="IT->Kimina distill", q17_it2klingspor="IT->20Q SFT", q17_it2coder_sft="IT->coder SFT")
add("4B", "it", q4_it="Qwen3-4B (official)", q4_inst2507="Instruct-2507", q4_think2507="Thinking-2507")
add("4B", "sft", q4_capsd_rand1k="SFT math 1k", q4_capsd_rand2k="SFT math 2k", q4_capsd_rand4k="SFT math 4k", q4_capsd_rand8k="SFT math 8k",
    q4_capsd_cap8k="SFT math 8k (cap)", q4_capsd_ppl8k="SFT math 8k (ppl)", q4_capsd_sci1k="SFT science 1k", q4_capsd_sci8k="SFT science 8k")
add("4B", "opd", q4_opsa="OPSA")
add("4B", "zero_rl", q4_grpo_ts="GRPO (thunlp)", q4_spiral_s0="Spiral s0", q4_spiral_s5="Spiral s5", q4_spiral_s10="Spiral s10", q4_spiral_s15="Spiral s15", q4_spiral_s22="Spiral s22")
add("4B", "rl_inc", q4_it2inst2507="IT->Instruct-2507", q4_it2think2507="IT->Thinking-2507", q4_it2polaris="IT->Polaris", q4_it2jannano="IT->Jan-nano (agentic)",
    q4_it2saferl="IT->SafeRL", q4_it2countdown="IT->Countdown RLVR", q4_it2nonthink_mathrl="IT->math RL")
add("4B", "sft_on_it", q4_it2calendar_sft="IT->calendar agent SFT", q4_it2devin_sft="IT->Devin SFT", q4_it2pythagoras="IT->prover SFT", q4_it2science_sft="IT->science SFT", q4_it2osim="IT->OSim", q4_it2guard="IT->Guard")
add("8B", "it", it="Qwen3-8B (official)")
add("8B", "sft", opd_sft3000="SFT OpenThoughts3 s3000", arm_sft100="SFT arm s100", klear_sft="Klear SFT", casc_sft="Cascade SFT", xcoder_sft="X-Coder SFT")
add("8B", "opd", opd_step200="OPD s200 (ours, after SFT)", arm_opd100="OPD arm s100", arm_opd25="OPD arm s25")
add("8B", "zero_rl", dapo27="DAPO s27", arm_rlvr500="RLVR arm s500", arm_rlvr100="RLVR arm s100")
add("8B", "rl_inc", klear_sft2rl="Klear SFT->RL", tmax_sft2rl8b="tmax SFT->RL", ota2_sft2rl="OTA2 SFT->RL", opd_sft2opd="SFT->OPD (ours)")
NL = {"1.7B": 28, "4B": 36, "8B": 36}
COL = {"it": "#5b5b5b", "sft": "#2a78d6", "sft_on_it": "#7fb0e6", "opd": "#1baf7a", "zero_rl": "#eda100", "rl_inc": "#eb6834", "dpo": "#9b59b6"}
KINDS = ("math", "code", "agent", "neutral")


def band_frac(nl):  # work band as depth fractions: 36 -> 20..26 ; 28 -> 15..20
    return (20, 26) if nl == 36 else (15, 20)


rows, prof = [], {}
for tag, (size, recipe, label) in T.items():
    f = f"{ZOO}/results/ro_{tag}_summary.json"
    if not os.path.exists(f): continue
    s = json.load(open(f)); L = s["layers"]; nl = NL[size]; lo, hi = band_frac(nl)
    for kind in KINDS:
        a = s["agg"].get(kind)
        if not a: continue
        ro0 = "J" if a.get("J") else "LL"
        js = {l: a[ro0][str(l)]["js"] for l in L}
        wb = float(np.mean([js[l] for l in L if lo <= l <= hi]))
        wbL = float(np.mean([a["LL"][str(l)]["js"] for l in L if lo <= l <= hi])) if a.get("LL") else None
        pk = max(js, key=js.get)
        rows.append({"tag": tag, "size": size, "recipe": recipe, "label": label, "kind": kind, "readout": ro0, "js_work": wb, "js_work_LL": wbL,
                     "js_final": a["final"]["js"], "peak_layer": pk, "peak_depth": (pk + 1) / nl, "js_peak": js[pk],
                     "nll_a": a.get("nll", {}).get("a"), "nll_b": a.get("nll", {}).get("b"), "n_layers": nl})
        if kind == "math": prof[tag] = ([(l + 1) / nl for l in L], [js[l] for l in L], size, recipe, label)
    loc = {r["kind"]: r["js_work"] for r in rows if r["tag"] == tag}
    for r in rows:
        if r["tag"] == tag and loc.get("neutral"): r["locality"] = r["js_work"] / loc["neutral"]

with open(f"{OUT}/size_axis.csv", "w", newline="") as f:
    keys = list(rows[0].keys()) + (["locality"] if "locality" not in rows[0] else [])
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

# console digest: math work-band write, locality, peak depth, grouped by size / recipe
print(f"{'size':5s} {'recipe':10s} {'tag':22s} {'label':30s} {'J work':>7s} {'LL work':>8s} {'final':>6s} {'loc':>5s} {'peak':>5s}")
for size in ("1.7B", "4B", "8B"):
    for recipe in COL:
        for r in sorted([r for r in rows if r["size"] == size and r["recipe"] == recipe and r["kind"] == "math"], key=lambda r: -r["js_work"]):
            ll = f"{r['js_work_LL']:8.3f}" if r["js_work_LL"] is not None else "       -"
            print(f"{size:5s} {recipe:10s} {r['tag']:22s} {r['label']:30s} {r['js_work']:7.3f} {ll} {r['js_final']:6.3f} {r.get('locality', float('nan')):5.2f} {r['peak_depth']:5.2f}")

# figure: math-probe J profiles vs depth, one panel per size
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, size in zip(axs, ("1.7B", "4B", "8B")):
        shown = set()
        for tag, (x, y, sz, recipe, label) in prof.items():
            if sz != size: continue
            lw = 2.0 if recipe in ("it", "sft", "opd", "zero_rl") else 1.0
            ax.plot(x, y, color=COL[recipe], lw=lw, alpha=0.9 if lw > 1 else 0.5, label=recipe if recipe not in shown else None)
            shown.add(recipe)
        lo, hi = band_frac(NL[size]); ax.axvspan((lo + 1) / NL[size], (hi + 1) / NL[size], color="0.92", zorder=0)
        ax.set_yscale("log"); ax.set_ylim(1e-3, 0.7); ax.set_xlabel("depth (layer / n_layers)"); ax.set_title(f"Qwen3-{size}", fontsize=10)
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7, frameon=False)
    axs[0].set_ylabel("J-lens JS vs parent (math probes)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_size_axis.pdf"); fig.savefig(f"{FIG}/fig7_size_axis.png", dpi=150); plt.close(fig)
    print(f"wrote {FIG}/fig7_size_axis.pdf")
except Exception as e:
    print("figure skipped:", e)
print(f"wrote {OUT}/size_axis.csv ({len(rows)} rows)")
