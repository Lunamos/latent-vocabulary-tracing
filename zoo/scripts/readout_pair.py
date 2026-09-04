"""Unified fixed-ruler readout for one (reference, comparison) model pair.

One HF forward per model per probe (output_hidden_states), then for every lens layer:
  J  : Jacobian lens  = unembed(h_l @ J_l^T)   (frozen lens J fitted on the reference base)
  LL : logit lens     = unembed(h_l)
and the model's own final logits ("final"). Metrics per position, exactly as opd/scripts/diff_lens.py:
  kl_ab, kl_ba, js, jaccard@10, top-50 ids/vals of both, cross logits (a_at_b, b_at_a), lse.
Plus hidden-state stats per layer (cos, linear CKA, dnorm_rel) as in opd/explore/hidden_cka.py, and
lens faithfulness KL(final_X || J_X), KL(final_X || LL_X) for X in {A, B}.

Conventions match jlens: block-L output = hidden_states[L+1]; residual fp32 -> @J^T (J cast to fp32)
-> cast to lm_head dtype -> final_norm -> lm_head -> fp32. Tokenization: tok(text, truncation, max_length=640).

Usage: python readout_pair.py MODEL_A MODEL_B TAG [--lens PATH] [--probes PATH] [--out DIR] [--kinds math,code,agent,neutral]
Outputs: OUT/ro_<TAG>.pt (store) and OUT/ro_<TAG>_summary.json
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
import transformers

ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(ZOO)
ap = argparse.ArgumentParser()
ap.add_argument("model_a"); ap.add_argument("model_b"); ap.add_argument("tag")
ap.add_argument("--lens", default=f"{ROOT}/repro/lenses/base_merged.pt")
ap.add_argument("--probes", default=f"{ZOO}/data/probes.jsonl")
ap.add_argument("--out", default=f"{ZOO}/results")
ap.add_argument("--kinds", default="math,code,agent,neutral")
ap.add_argument("--lmin", type=int, default=6); ap.add_argument("--lmax", type=int, default=34)
ap.add_argument("--topk", type=int, default=10); ap.add_argument("--max_len", type=int, default=640)
ap.add_argument("--no_store", action="store_true", help="only write the summary json")
ap.add_argument("--full_LL", action="store_true", help="store LL cross logits too (top-k, a_at_b/b_at_a) instead of the slim top-20 store")
ap.add_argument("--no_J", action="store_true", help="no Jacobian lens for this architecture: logit lens + final + hidden only")
ap.add_argument("--layers", default="", help="explicit layer list (e.g. 4,5,...,30) when --no_J; default = lens layers")
args = ap.parse_args()
dev = "cuda:0"
t0 = time.time()

if args.no_J:
    lens = {"n_prompts": 0}
    LAYERS = [int(x) for x in args.layers.split(",")] if args.layers else list(range(args.lmin, args.lmax + 1))
    J = None
    print(f"no J-lens; layers {LAYERS[0]}..{LAYERS[-1]} ({len(LAYERS)})", flush=True)
else:
    lens = torch.load(args.lens, map_location="cpu", weights_only=True)
    LAYERS = [l for l in lens["source_layers"] if args.lmin <= l <= args.lmax]
    J = {l: lens["J"][l].to(dev, torch.float32) for l in LAYERS}
    print(f"lens {args.lens} n_prompts={lens['n_prompts']} layers {LAYERS[0]}..{LAYERS[-1]} ({len(LAYERS)})", flush=True)
READOUTS = ("LL",) if args.no_J else ("J", "LL")


def load(name):
    m = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev).eval()
    return m


tok = transformers.AutoTokenizer.from_pretrained(args.model_a)
model_a, model_b = load(args.model_a), load(args.model_b)
tok_b = transformers.AutoTokenizer.from_pretrained(args.model_b)
# Protocol: every model is read on the SAME token ids (tokenized by the reference tokenizer). Only require that
# the two tokenizers agree on the id of every shared token (some descendants re-register chat specials, e.g. R1-distill).
va, vb = tok.get_vocab(), tok_b.get_vocab()
bad = [k for k in va if k in vb and va[k] != vb[k]]
if bad:   # tolerate remapped special/extra tokens as long as none of them occurs in the probes
    bad_ids = {va[k] for k in bad} | {vb[k] for k in bad}
    _probe_ids = set()
    for _p in (json.loads(l) for l in open(args.probes)):
        _probe_ids |= set(tok(_p["text"], truncation=True, max_length=args.max_len).input_ids)
    hit = sorted(bad_ids & _probe_ids)
    assert not hit, f"tokenizer id mismatch on tokens present in probes: {hit[:5]}"
    print(f"WARNING: {len(bad)} shared tokens remapped between tokenizers (e.g. {bad[:4]}); none occur in probes -> continuing", flush=True)
assert model_a.config.vocab_size == model_b.config.vocab_size, "vocab_size mismatch"
tok_note = {"shared_tokens": len(set(va) & set(vb)), "only_a": sorted(set(va) - set(vb))[:20], "only_b": sorted(set(vb) - set(va))[:20],
            "b_rope": str(getattr(model_b.config, "rope_scaling", None) or getattr(model_b.config, "rope_parameters", None))}
print(f"models loaded ({time.time()-t0:.0f}s) tokenizer note: {tok_note}", flush=True)


def _parts(m):
    inner = getattr(m, "model", None) or getattr(m, "transformer", None)
    # VL / multimodal wrappers keep the text stack one level deeper
    for sub in ("language_model", "text_model"):
        if hasattr(inner, sub): inner = getattr(inner, sub); break
    norm = getattr(inner, "norm", None) or getattr(inner, "final_layer_norm", None) or getattr(inner, "ln_f", None)
    head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None)
    assert norm is not None and head is not None, "cannot locate final norm / lm_head"
    return norm, head


def unembed(m, x):
    norm, head = _parts(m)
    return head(norm(x.to(head.weight.dtype))).float()


def pair_metrics(A, B, topk):
    """A,B: [pos, vocab] fp32 logits on GPU -> dict of per-position tensors (cpu)."""
    lpa, lpb = F.log_softmax(A, -1), F.log_softmax(B, -1)
    pa, pb = lpa.exp(), lpb.exp()
    z0 = torch.zeros((), device=A.device)
    kl_ab = torch.where(pa > 0, pa * (lpa - lpb), z0).sum(-1)
    kl_ba = torch.where(pb > 0, pb * (lpb - lpa), z0).sum(-1)
    lm = torch.logaddexp(lpa, lpb) - 0.6931471805599453   # log of the mixture, no underflow
    z = torch.zeros((), device=A.device)
    js = 0.5 * (torch.where(pa > 0, pa * (lpa - lm), z).sum(-1) + torch.where(pb > 0, pb * (lpb - lm), z).sum(-1))
    va, ta = A.topk(50, -1)
    vb, tb = B.topk(50, -1)
    # jaccard@topk vectorized
    inter = (ta[:, :topk, None] == tb[:, None, :topk]).any(-1).sum(-1).float()
    jac = inter / (2 * topk - inter)
    out = {"kl_ab": kl_ab, "kl_ba": kl_ba, "js": js, "jaccard": jac,
           "top_a": ta.int(), "top_b": tb.int(), "val_a": va, "val_b": vb,
           "a_at_b": torch.gather(A, 1, tb), "b_at_a": torch.gather(B, 1, ta),
           "lse_a": A.logsumexp(-1), "lse_b": B.logsumexp(-1)}
    return out


def to_cpu_half(d):
    return {k: (v.cpu().half() if v.dtype == torch.float32 else v.cpu()) for k, v in d.items()}


def kl(P_logits, Q_logits):
    lp, lq = F.log_softmax(P_logits, -1), F.log_softmax(Q_logits, -1)
    p = lp.exp()
    return torch.where(p > 0, p * (lp - lq), torch.zeros((), device=p.device)).sum(-1)


kinds = set(args.kinds.split(","))
probes = [json.loads(l) for l in open(args.probes)]
probes = [p for p in probes if p["kind"] in kinds]
print(f"{len(probes)} probes, kinds={sorted(kinds)}", flush=True)

records, store = [], {}
with torch.no_grad():
    for i, p in enumerate(probes):
        ids = tok(p["text"], return_tensors="pt", truncation=True, max_length=args.max_len).input_ids
        ids = ids.to(dev)
        oa = model_a(input_ids=ids, output_hidden_states=True, use_cache=False)
        ob = model_b(input_ids=ids, output_hidden_states=True, use_cache=False)
        fa, fb = oa.logits[0].float(), ob.logits[0].float()
        n_pos = ids.shape[1]
        rec = {"key": p["key"], "kind": p["kind"], "prompt_len": p["prompt_len"], "meta": p["meta"],
               "n_pos": n_pos, "per_layer": {"J": {}, "LL": {}}, "hidden": {}, "faith": {"J": {}, "LL": {}}}
        st = {"n_pos": n_pos, "input_ids": ids[0].cpu().int(), "J": {}, "LL": {}, "store_offset": p["prompt_len"]}
        span = slice(p["prompt_len"], n_pos)  # response span (the per-position store keeps ONLY this span)
        for l in LAYERS:
            ha, hb = oa.hidden_states[l + 1][0].float(), ob.hidden_states[l + 1][0].float()
            # hidden stats
            cos = float(F.cosine_similarity(ha, hb, dim=-1).mean())
            Xc, Yc = ha - ha.mean(0, keepdim=True), hb - hb.mean(0, keepdim=True)
            K, L2 = (Xc @ Xc.T).double(), (Yc @ Yc.T).double()
            cka = float((K * L2).sum() / (K.norm() * L2.norm() + 1e-12))
            dnr = float((hb - ha).norm(dim=-1).mean() / (ha.norm(dim=-1).mean() + 1e-12))
            rec["hidden"][str(l)] = {"cos": cos, "cka": cka, "dnorm_rel": dnr}
            for kind, xa, xb in ((("J", ha @ J[l].T, hb @ J[l].T),) if J is not None else ()) + (("LL", ha, hb),):
                A, B = unembed(model_a, xa), unembed(model_b, xb)
                m = pair_metrics(A, B, args.topk)
                rec["per_layer"][kind][str(l)] = {
                    k: float(m[k].mean()) for k in ("kl_ab", "kl_ba", "js", "jaccard")} | {
                    k + "_resp": float(m[k][span].mean()) if n_pos > p["prompt_len"] else None
                    for k in ("kl_ab", "kl_ba", "js")}
                rec["faith"][kind][str(l)] = {"a": float(kl(fa, A).mean()), "b": float(kl(fb, B).mean())}
                if not args.no_store:
                    if kind == "LL" and not args.full_LL:   # slim store: top-20, no cross logits (disk budget)
                        m = {k: (v[:, :20] if k in ("top_a", "top_b", "val_a", "val_b") else v)
                             for k, v in m.items() if k not in ("a_at_b", "b_at_a")}
                    st[kind][l] = to_cpu_half({k: v[span] for k, v in m.items()})
                del A, B, m
        # final logits
        m = pair_metrics(fa, fb, args.topk)
        rec["final"] = {k: float(m[k].mean()) for k in ("kl_ab", "kl_ba", "js", "jaccard")} | {
            k + "_resp": float(m[k][span].mean()) if n_pos > p["prompt_len"] else None
            for k in ("kl_ab", "kl_ba", "js")}
        if not args.no_store:
            st["final"] = to_cpu_half({k: v[span] for k, v in m.items()})
        # per-token logprob of the actual next token (teacher-forced NLL) for both models on the response span
        tgt = ids[0, 1:]
        nll_a = -F.log_softmax(fa[:-1], -1).gather(1, tgt[:, None])[:, 0]
        nll_b = -F.log_softmax(fb[:-1], -1).gather(1, tgt[:, None])[:, 0]
        rs = slice(max(p["prompt_len"] - 1, 0), n_pos - 1)
        rec["nll"] = {"a": float(nll_a[rs].mean()), "b": float(nll_b[rs].mean())}
        st["nll_a"], st["nll_b"] = nll_a.cpu().half(), nll_b.cpu().half()
        records.append(rec)
        store[p["key"]] = st
        del oa, ob, fa, fb, m
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(probes)} ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(args.out, exist_ok=True)
# aggregate summary
agg = {}
for kind in sorted(kinds):
    rs = [r for r in records if r["kind"] == kind]
    if not rs:
        continue
    a = {"n": len(rs), "J": {}, "LL": {}, "hidden": {}, "faith": {"J": {}, "LL": {}}}
    for l in LAYERS:
        for ro in READOUTS:
            a[ro][str(l)] = {k: sum(r["per_layer"][ro][str(l)][k] for r in rs) / len(rs)
                             for k in ("kl_ab", "kl_ba", "js", "jaccard")}
            for metric in ("kl_ab", "kl_ba", "js"):
                key = f"{metric}_resp"
                vals = [
                    r["per_layer"][ro][str(l)][key]
                    for r in rs
                    if r["per_layer"][ro][str(l)][key] is not None
                ]
                a[ro][str(l)][key] = sum(vals) / len(vals) if vals else None
            a["faith"][ro][str(l)] = {x: sum(r["faith"][ro][str(l)][x] for r in rs) / len(rs) for x in ("a", "b")}
        a["hidden"][str(l)] = {k: sum(r["hidden"][str(l)][k] for r in rs) / len(rs) for k in ("cos", "cka", "dnorm_rel")}
    a["final"] = {k: sum(r["final"][k] for r in rs) / len(rs) for k in ("kl_ab", "kl_ba", "js", "jaccard")}
    for metric in ("kl_ab", "kl_ba", "js"):
        key = f"{metric}_resp"
        vals = [r["final"][key] for r in rs if r["final"][key] is not None]
        a["final"][key] = sum(vals) / len(vals) if vals else None
    a["nll"] = {x: sum(r["nll"][x] for r in rs) / len(rs) for x in ("a", "b")}
    agg[kind] = a
summary = {"model_a": args.model_a, "model_b": args.model_b, "tag": args.tag, "lens": args.lens,
           "decoder_mode": "native_per_model",
           "lens_n_prompts": int(lens["n_prompts"]), "readouts": list(READOUTS), "layers": LAYERS, "probes": args.probes,
           "seconds": time.time() - t0, "tokenizer_note": tok_note, "agg": agg, "records": records}
json.dump(summary, open(os.path.join(args.out, f"ro_{args.tag}_summary.json"), "w"), indent=1)
if not args.no_store:
    torch.save({"records": records, "store": store, "layers": LAYERS, "model_a": args.model_a,
                "model_b": args.model_b, "lens": args.lens}, os.path.join(args.out, f"ro_{args.tag}.pt"))
# one-line digest
def dig(kind):
    a = agg.get(kind)
    if not a: return ""
    ro = "J" if "J" in READOUTS else "LL"
    wb = [a[ro][str(l)]["js"] for l in LAYERS if 20 <= l <= 26] or [a[ro][str(LAYERS[len(LAYERS)//2])]["js"]]
    return f"{kind}: {ro}-work-band JS={sum(wb)/len(wb):.4f} final JS={a['final']['js']:.4f} L22 cos={a['hidden']['22']['cos']:.3f} nll a/b={a['nll']['a']:.3f}/{a['nll']['b']:.3f}"
print(f"DONE {args.tag} in {time.time()-t0:.0f}s | " + " | ".join(dig(k) for k in sorted(kinds)), flush=True)
