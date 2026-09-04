"""What is the shared RL-from-base direction, in words?  And is it rank-targeted rather than proportional sharpening?

For each tag: on A's (Base) top-50 support at J work band (L20-26), response span, math probes:
  (1) mean delta-logprob by A-rank bucket (1, 2-5, 6-20, 21-50): does B mostly move mid-rank tokens?
  (2) per-token mean delta (as content_table, A-support) -> top suppressed/promoted lists
Then across the given tags: intersection of suppressed / promoted top-N lists and a pooled direction
(average of per-token means over tags, tokens present in all) -> printed as the "consensus" table.
Usage: python rl_direction.py TAG [TAG ...] [--kind math] [--band 20-26] [--topn 40]
"""
import sys, re, argparse, json, os
import numpy as np, torch, transformers
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"
ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--kind", default="math")
ap.add_argument("--band", default="20-26"); ap.add_argument("--topn", type=int, default=40); ap.add_argument("--min_n", type=int, default=20)
args = ap.parse_args()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base"); V = len(tok)
lo, hi = map(int, args.band.split("-"))
BUCKETS = [(0, 1, "rank1"), (1, 5, "rank2-5"), (5, 20, "rank6-20"), (20, 50, "rank21-50")]
per_tok = {}
print(f"{'tag':14s} " + " ".join(f"{b[2]:>10s}" for b in BUCKETS) + "   (mean delta logprob on Base support, J L%s, %s)" % (args.band, args.kind))
for tag in args.tags:
    d = torch.load(f"{ZOO}/results/ro_{tag}.pt", map_location="cpu", weights_only=False)
    layers = [l for l in d["layers"] if lo <= l <= hi]
    sums, cnts = np.zeros(V), np.zeros(V, dtype=np.int64); bsum = np.zeros(len(BUCKETS)); bcnt = np.zeros(len(BUCKETS))
    for rec in d["records"]:
        if rec["kind"] != args.kind: continue
        st = d["store"][rec["key"]]
        for l in layers:
            L = st["J"][l]
            ta = L["top_a"].long().numpy(); va = L["val_a"].float().numpy(); lsa = L["lse_a"].float().numpy()[:, None]
            baa = L["b_at_a"].float().numpy(); lsb = L["lse_b"].float().numpy()[:, None]
            dlt = (baa - lsb) - (va - lsa)          # [pos, 50], columns are A-rank order (topk sorted)
            if dlt.shape[0] == 0: continue
            np.add.at(sums, ta.ravel(), dlt.ravel()); np.add.at(cnts, ta.ravel(), 1)
            for i, (a, b, _) in enumerate(BUCKETS):
                bsum[i] += dlt[:, a:b].sum(); bcnt[i] += dlt[:, a:b].size
    mean = np.where(cnts >= args.min_n, sums / np.maximum(cnts, 1), np.nan); per_tok[tag] = mean
    print(f"{tag:14s} " + " ".join(f"{bsum[i]/bcnt[i]:+10.3f}" for i in range(len(BUCKETS))))
CJK = re.compile(r"[一-鿿]")
def top(mean, n, sign):
    ok = np.where(~np.isnan(mean))[0]; order = ok[np.argsort(mean[ok])]
    return list(order[:n]) if sign < 0 else list(order[::-1][:n])
lists = {t: (set(top(m, args.topn, -1)), set(top(m, args.topn, +1))) for t, m in per_tok.items()}
tags = args.tags
if len(tags) > 1:
    print("\npairwise overlap of top-%d suppressed / promoted token sets:" % args.topn)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            s1, p1 = lists[tags[i]]; s2, p2 = lists[tags[j]]
            print(f"  {tags[i]:14s} ~ {tags[j]:14s}: suppressed {len(s1 & s2):2d}/{args.topn}  promoted {len(p1 & p2):2d}/{args.topn}")
    M = np.vstack([per_tok[t] for t in tags]); common = ~np.isnan(M).any(0); pooled = np.where(common, M.mean(0), np.nan)
    sup, pro = top(pooled, args.topn, -1), top(pooled, args.topn, +1)
    print("\nCONSENSUS direction (mean over tags; tokens scored in all): SUPPRESSED | PROMOTED")
    for a, b in zip(sup, pro):
        print(f"  {pooled[a]:+6.2f} {tok.decode([int(a)])!r:22s} | {pooled[b]:+6.2f} {tok.decode([int(b)])!r}")
    print(f"CJK share: suppressed {np.mean([bool(CJK.search(tok.decode([int(a)]))) for a in sup]):.2f} promoted {np.mean([bool(CJK.search(tok.decode([int(b)]))) for b in pro]):.2f}")
    os.makedirs(f"{ZOO}/analysis/content", exist_ok=True)
    json.dump({"tags": tags, "kind": args.kind, "band": args.band, "suppressed": [(int(a), tok.decode([int(a)]), float(pooled[a])) for a in sup],
               "promoted": [(int(b), tok.decode([int(b)]), float(pooled[b])) for b in pro]}, open(f"{ZOO}/analysis/content/consensus_{'_'.join(tags)}_{args.kind}.json", "w"), ensure_ascii=False, indent=1)
