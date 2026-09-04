"""Vocabulary taxonomy of what a descendant writes into the workspace (probability-mass version).

For each (position, layer) in the work band, on model A's top-50 support, we take p_A and p_B of each support token
(B's probability of A's tokens, from the cross-logits). Mass moved up = sum max(p_B-p_A,0), moved down = sum max(p_A-p_B,0). Every vocabulary token is assigned to one category (special / cjk / other_script / junk /
format / punct / number / math / code / discourse / function / english / latin_piece / other) and we report, per tag:
  base composition  : share of A-support occurrences per category (what the parent's workspace is made of)
  promoted mass     : share of sum(max(delta,0)) per category   (where the write adds probability)
  suppressed mass   : share of sum(max(-delta,0)) per category  (where the write removes probability)
  net per category  : mean delta over occurrences of that category (nats)
Outputs: zoo/analysis/taxonomy/<tag>_<readout>_<kind>_<band>.json, taxonomy.csv (append/replace rows),
         zoo/figs/wc_<tag>_<readout>_<kind>.png  (word cloud: promoted in green, suppressed in red, size = mass)
Usage: python write_taxonomy.py TAG [TAG ...] [--readout J|LL|final] [--kind math] [--band 20-26] [--no_cloud]
       band default is chosen from the store's layer count (36-layer -> 20-26, 28-layer -> 15-20, 32-layer -> 14-20).
"""
import argparse, json, os, re, csv, unicodedata
import numpy as np, torch
import transformers
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ZOO}/analysis/taxonomy"; FIG = f"{ZOO}/figs"; os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("tags", nargs="+"); ap.add_argument("--readout", default="J"); ap.add_argument("--kind", default="math")
ap.add_argument("--band", default=""); ap.add_argument("--no_cloud", action="store_true"); ap.add_argument("--topk_cloud", type=int, default=200)
args = ap.parse_args()

# ---------------- token categories ----------------
CATS = ["special", "cjk", "other_script", "junk", "junk_id", "format", "punct", "number", "math", "code", "discourse", "function", "english", "latin_piece", "other"]
DISCOURSE = set("""wait alternatively so therefore thus hmm okay ok but first second next then now let lets let's actually however since because hence
verify check recall note suppose assume indeed clearly obviously right yes no maybe perhaps finally overall wait, hmm, okay, alright well also again
double-check confirm conclusion answer step steps approach method idea consider try let’s""".split())
FUNCTION = set("""the a an of to in and or is are was were be been being that this these those it its for on with as by at from we i you they he she
him her his their our your my me us them can could will would shall should may might must not do does did done have has had having which what who whom
whose where when why how if than then there here into onto out over under up down about above below between through during before after while
all any each every some such only own same other another more most less least very too just also than each both either neither nor yet still already
one two three four five six seven eight nine ten""".split())
CODE_KW = set("""def return import from class self print int str float bool list dict set tuple None True False for while if elif else try except
finally with lambda yield pass break continue raise assert global nonlocal async await const let var function new this null undefined void static
public private protected struct enum typedef include using namespace std cout cin endl nullptr auto template virtual override switch case default
do goto sizeof unsigned char long short double main args argv len range append extend sorted sort map filter reduce zip enumerate isinstance
input output printf scanf malloc free string vector array object json math random numpy np pd torch tf""".split())
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
OTHER_SCRIPT_RE = re.compile(r"[Ͱ-ϿЀ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿ᄀ-ᇿ]")
GREEK_RE = re.compile(r"[Ͱ-Ͽ]")
MATH_RE = re.compile(r"(\\[a-zA-Z]+|^\s*[=+\-*/^≤≥≠±×÷∑∏∫√∞∂∇∈∉⊂⊆∪∩→⇒⇔∀∃∘·$%]+\s*$|boxed|\bfrac\b|\bsqrt\b|\bcdot\b|\btimes\b|\btheta\b|\balpha\b|\bbeta\b|\bgamma\b|\blambda\b|\bsigma\b|\bpi\b)")
NUM_RE = re.compile(r"^\s*[-+]?\d[\d,.]*\s*$")
CODE_OP_RE = re.compile(r"(==|!=|<=|>=|\+=|-=|\*=|/=|=>|->|::|&&|\|\||//|/\*|\*/|#!|\(\)|\[\]|\{\}|;\s*$|^\s*(#include|#define|import|from)\b)")
FORMAT_RE = re.compile(r"^\s*$|^[\s*#`>|_=\-~]+$|^\s*(\*\*|###|##|#|---|```|\\n)+\s*$")
PUNCT_RE = re.compile(r"^\s*[!\"'(),.:;?\[\]{}<>/\\|@&#~`«»“”‘’…—–、。，：；！？（）「」『』]+\s*$")
WORD_RE = re.compile(r"^\s?[A-Za-z]+[’']?[A-Za-z]*$")


def categorize(t: str) -> str:
    s = t
    if "<|" in s or "<think" in s or "</think" in s or "<tool" in s or "</tool" in s: return "special"
    if "�" in s or any(unicodedata.category(c) == "Cc" and c not in "\n\t\r" for c in s): return "junk"
    if CJK_RE.search(s): return "cjk"
    if GREEK_RE.search(s): return "math"
    if OTHER_SCRIPT_RE.search(s): return "other_script"
    if FORMAT_RE.match(s): return "format"
    if PUNCT_RE.match(s): return "punct"
    if NUM_RE.match(s): return "number"
    if MATH_RE.search(s) and not WORD_RE.match(s): return "math"
    st = s.strip()
    if re.match(r"^[.(\[{]?[A-Za-z_][A-Za-z0-9_]*$", st) and len(st) >= 10 and (("_" in st) or len(re.findall(r"[A-Z]", st[1:])) >= 2): return "junk_id"
    if CODE_OP_RE.search(s) or st.lower() in CODE_KW or ("_" in st and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", st)) or re.match(r"^[a-z]+[A-Z][A-Za-z]*$", st) \
       or (len(re.findall(r"[A-Z]", st[1:])) >= 2 and re.match(r"^[A-Za-z]+$", st)) or re.match(r"^[.(\[{]\s*[A-Za-z_][A-Za-z0-9_]*$", st): return "code"
    low = st.lower().rstrip(",.:;")
    if WORD_RE.match(s):
        if low in DISCOURSE or low.replace("’", "'") in DISCOURSE: return "discourse"
        if low in FUNCTION: return "function"
        if not s.startswith(" ") and len(st) <= 3 and st.islower(): return "latin_piece"
        if s.startswith(" ") or st[:1].isupper() or len(st) >= 4: return "english"
        return "latin_piece"
    if re.match(r"^\s?[A-Za-z]", s): return "latin_piece"
    if "\\" in s or "$" in s or MATH_RE.search(s): return "math"        # LaTeX-ish mixtures: ')\\', '.$', '+\\'
    if "**" in s or "#" in s or "`" in s or "|" in s: return "format"     # markdown mixtures: ',**', ':**'
    if re.match(r"^\s*[-–—]+[A-Za-z]*$", s) or re.match(r"^[^\w\s]+$", s): return "punct"
    return "other"


# ---------------- per-tag aggregation ----------------
def band_for(layers):
    if args.band: lo, hi = map(int, args.band.split("-")); return [l for l in layers if lo <= l <= hi]
    m = max(layers)
    lo, hi = (20, 26) if m >= 33 else ((14, 20) if m >= 29 else (15, 20))
    return [l for l in layers if lo <= l <= hi]


def analyse(tag):
    d = torch.load(f"{ZOO}/results/ro_{tag}.pt", map_location="cpu", weights_only=False)
    mm = re.search(r"models--([^/]+)--([^/]+)/snapshots", d["model_a"])
    tok = transformers.AutoTokenizer.from_pretrained(f"{mm.group(1)}/{mm.group(2)}" if mm else d["model_a"])
    ro = args.readout
    st0 = d["store"][d["records"][0]["key"]]
    if ro == "J" and not st0.get("J"): ro = "LL"
    layers = ["final"] if ro == "final" else band_for(d["layers"])
    V = len(tok) + 512
    PA, PB, up, down, cnt = np.zeros(V), np.zeros(V), np.zeros(V), np.zeros(V), np.zeros(V, dtype=np.int64)
    n_rec = n_pos = 0
    for rec in d["records"]:
        if rec["kind"] != args.kind: continue
        st = d["store"][rec["key"]]
        p_lo, p_hi = rec["prompt_len"], st["n_pos"]
        if p_hi <= p_lo: continue
        n_rec += 1; n_pos += p_hi - p_lo
        off = st.get("store_offset", 0); p_lo, p_hi = p_lo - off, p_hi - off
        for l in layers:
            L = st["final"] if l == "final" else st[ro][l]
            if "b_at_a" not in L: continue   # slim store: skip
            ta = L["top_a"].long().numpy()[p_lo:p_hi]; va = L["val_a"].float().numpy()[p_lo:p_hi]
            lsa = L["lse_a"].float().numpy()[p_lo:p_hi, None]; lsb = L["lse_b"].float().numpy()[p_lo:p_hi, None]
            baa = L["b_at_a"].float().numpy()[p_lo:p_hi]
            pa = np.exp(va - lsa); pb = np.exp(baa - lsb)          # probabilities of A's top-k tokens under A and under B
            ids = ta.ravel(); pa = pa.ravel(); pb = pb.ravel()
            np.add.at(PA, ids, pa); np.add.at(PB, ids, pb)
            np.add.at(up, ids, np.maximum(pb - pa, 0)); np.add.at(down, ids, np.maximum(pa - pb, 0)); np.add.at(cnt, ids, 1)
    ok = np.nonzero(cnt)[0]
    cat_of = {int(t): categorize(tok.decode([int(t)])) for t in ok}
    comp = {c: 0.0 for c in CATS}; pm = {c: 0.0 for c in CATS}; nm = {c: 0.0 for c in CATS}; pbm = {c: 0.0 for c in CATS}; ncat = {c: 0 for c in CATS}
    for t in ok:
        c = cat_of[int(t)]; comp[c] += float(PA[t]); pbm[c] += float(PB[t]); pm[c] += float(up[t]); nm[c] += float(down[t]); ncat[c] += int(cnt[t])
    tot_c, tot_p, tot_n = sum(comp.values()), sum(pm.values()), sum(nm.values())
    res = {"tag": tag, "model_a": d["model_a"], "model_b": d["model_b"], "readout": ro, "layers": [str(l) for l in layers], "kind": args.kind,
           "n_records": n_rec, "n_positions": n_pos * len(layers), "n_tokens": int(len(ok)),
           "total_promoted_mass": tot_p, "total_suppressed_mass": tot_n, "mass_per_occurrence": (tot_p + tot_n) / max(n_pos * len(layers), 1),
           "support_mass_A": tot_c / max(n_pos * len(layers), 1),
           "cross_category_share": (sum(abs(pm[c] - nm[c]) for c in CATS) / 2) / max(tot_p + tot_n, 1e-9) * 2,
           "cats": {c: {"comp_share": comp[c] / max(tot_c, 1e-9), "promoted_share": pm[c] / max(tot_p, 1e-9), "suppressed_share": nm[c] / max(tot_n, 1e-9),
                        "net_mean": (pbm[c] - comp[c]) / max(comp[c], 1e-9), "n": ncat[c]} for c in CATS}}
    # top tokens for the cloud: mass = pos_sum or neg_sum; keep sign by dominant side
    top_p = sorted(((float(up[t] - down[t]), int(t)) for t in ok if up[t] > down[t]), reverse=True)[:args.topk_cloud]
    top_n = sorted(((float(down[t] - up[t]), int(t)) for t in ok if down[t] > up[t]), reverse=True)[:args.topk_cloud]
    res["top_promoted"] = [{"tok": tok.decode([t]), "mass": m, "n": int(cnt[t]), "pA": float(PA[t]), "pB": float(PB[t]), "cat": cat_of[t]} for m, t in top_p]
    res["top_suppressed"] = [{"tok": tok.decode([t]), "mass": m, "n": int(cnt[t]), "pA": float(PA[t]), "pB": float(PB[t]), "cat": cat_of[t]} for m, t in top_n]
    return res


def cloud(res, path, content_only=False):
    from wordcloud import WordCloud
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    font = os.path.expanduser("~/.local/share/fonts/NotoSansSC-Regular.otf")
    if not os.path.exists(font): font = None
    SKIP = {"punct", "format", "junk", "junk_id", "special", "other"} if content_only else set()
    def show(t):  # visible token text
        s = t.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
        if s.startswith(" "): s = "_" + s[1:]
        return s if s.strip() else "_"
    def name(pth):
        m = re.search(r"models--([^/]+)--([^/]+)/snapshots", pth); return (m.group(2) if m else pth.split("/")[-1])[:45]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, col, title in ((axs[0], "top_promoted", "#1b7f3a", "promoted (descendant gains)"), (axs[1], "top_suppressed", "#b22222", "suppressed (descendant loses)")):
        freqs = {}
        for r in res[key]:
            if r["cat"] in SKIP: continue
            k = show(r["tok"]); freqs[k] = freqs.get(k, 0) + max(r["mass"], 1e-6)
        if not freqs: ax.axis("off"); continue
        wc = WordCloud(width=900, height=520, background_color="white", font_path=font, prefer_horizontal=1.0, color_func=lambda *a, **k: col,
                       max_words=args.topk_cloud, relative_scaling=0.6, collocations=False, regexp=r".+", min_font_size=6).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(title, fontsize=11)
    fig.suptitle(f"{res['tag']}: {name(res['model_a'])} -> {name(res['model_b'])} · {res['readout']} L{res['layers'][0]}-{res['layers'][-1]} · {res['kind']} probes"
                 + (" · content tokens only" if content_only else " · all tokens") + " · size = probability mass moved", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

rows = []
for tag in args.tags:
    try: res = analyse(tag)
    except Exception as e: print(f"{tag}: skipped ({type(e).__name__}: {str(e)[:80]})"); continue
    band = f"{res['layers'][0]}-{res['layers'][-1]}"
    json.dump(res, open(f"{OUT}/{tag}_{res['readout']}_{args.kind}_{band}.json", "w"), indent=1, ensure_ascii=False)
    if not args.no_cloud:
        cloud(res, f"{FIG}/wc_{tag}_{res['readout']}_{args.kind}.png"); cloud(res, f"{FIG}/wcc_{tag}_{res['readout']}_{args.kind}.png", content_only=True)
    c = res["cats"]
    rows.append({"tag": tag, "readout": res["readout"], "kind": args.kind, "band": band, "mass_per_occ": res["mass_per_occurrence"], "cross_share": res["cross_category_share"],
                 **{f"comp_{k}": v["comp_share"] for k, v in c.items()}, **{f"pro_{k}": v["promoted_share"] for k, v in c.items()},
                 **{f"sup_{k}": v["suppressed_share"] for k, v in c.items()}, **{f"net_{k}": v["net_mean"] for k, v in c.items()}})
    print(f"\n== {tag} ({res['readout']} L{band}, {args.kind}; {res['n_positions']} (pos,layer) cells; A-support mass {res['support_mass_A']:.2f}; moved {res['mass_per_occurrence']:.3f} prob/cell)")
    print(f"{'category':13s} {'base%':>6s} {'gain%':>6s} {'loss%':>6s} {'net chg':>8s}   top promoted | top suppressed")
    for k in CATS:
        v = c[k]
        if v["n"] == 0: continue
        tp = ", ".join(repr(r["tok"]) for r in res["top_promoted"] if r["cat"] == k)[:60]
        ts = ", ".join(repr(r["tok"]) for r in res["top_suppressed"] if r["cat"] == k)[:60]
        print(f"{k:13s} {100*v['comp_share']:6.1f} {100*v['promoted_share']:6.1f} {100*v['suppressed_share']:6.1f} {100*v['net_mean']:+6.0f}%   {tp} | {ts}")
# merge into taxonomy.csv (replace rows with same tag/readout/kind)
path = f"{OUT}/taxonomy.csv"; old = []
if os.path.exists(path):
    old = [r for r in csv.DictReader(open(path)) if not any(r["tag"] == n["tag"] and r["readout"] == n["readout"] and r["kind"] == n["kind"] for n in rows)]
allrows = old + rows
if allrows:
    keys = list(rows[0].keys()) if rows else list(allrows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(allrows)
print(f"\nwrote {path} ({len(allrows)} rows)")
