"""Standardized behavior axis for the zoo (vLLM). Same prompts/sampling for every model; each model uses its
OWN chat template (thinking enabled where the template supports it), because that is how the recipe is meant
to be used. Benchmarks: AIME24+AIME25 (60 problems, avg@K) and MATH-500 (avg@1). Scoring = math_verify.

Usage: CUDA_VISIBLE_DEVICES=g python eval_zoo.py MODEL_PATH TAG [--k 4] [--max_tokens 16384] [--math500 1]
Output: zoo/results/eval_<tag>.json
"""
import argparse, json, os, sys, time
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); ROOT = os.path.dirname(ZOO)
os.environ.setdefault("HF_HOME", "/localscratch/zjin350/hf_cache")
sys.path.insert(0, f"{ROOT}/opd/scripts")
from opd_common import SYSTEM_MATH, check_answer
from datasets import load_dataset
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--k", type=int, default=2)
ap.add_argument("--max_tokens", type=int, default=12288); ap.add_argument("--math500", type=int, default=1)
ap.add_argument("--max_model_len", type=int, default=20480); ap.add_argument("--no_think", action="store_true")
ap.add_argument("--n_math", type=int, default=100, help="MATH-500 subset size (seeded shuffle); 500 = full")
def main():
    args = ap.parse_args()
    OUT = f"{ZOO}/results"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)


    def render(problem):
        msgs = [{"role": "system", "content": SYSTEM_MATH}, {"role": "user", "content": problem}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=not args.no_think)
        except Exception:
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return f"{SYSTEM_MATH}\n\nProblem: {problem}\n\nSolution:"


    items = []
    for name, hid, pcol, acol in [("AIME24", "Maxwell-Jia/AIME_2024", "Problem", "Answer"),
                                  ("AIME25", "yentinglin/aime_2025", "problem", "answer")]:
        ds = load_dataset(hid); ds = ds[list(ds.keys())[0]]
        for r in ds:
            items.append({"bench": name, "prompt": render(str(r[pcol])), "gold": str(r[acol]), "k": args.k})
    if args.math500:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        if args.n_math < len(ds): ds = ds.shuffle(seed=0).select(range(args.n_math))
        for r in ds:
            items.append({"bench": "MATH500", "prompt": render(r["problem"]), "gold": r["answer"], "k": 1})
    print(f"{len(items)} items; example prompt tail: {items[0]['prompt'][-160:]!r}", flush=True)

    from vllm import LLM, SamplingParams
    import subprocess
    gid = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    q = subprocess.check_output(["nvidia-smi", "-i", gid, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"]).decode().split(",")
    free_b, total_b = float(q[0]) * 2**20, float(q[1]) * 2**20   # no CUDA init in the parent (vLLM spawns workers)
    util = min(0.88, (free_b - 3 * 2**30) / total_b)   # share the GPU with whatever else is running
    assert util > 0.28, f"not enough free GPU memory ({free_b/2**30:.1f} GiB)"
    print(f"gpu free {free_b/2**30:.1f}/{total_b/2**30:.1f} GiB -> gpu_memory_utilization={util:.2f}", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=args.max_model_len,
              tensor_parallel_size=1, seed=0, enable_prefix_caching=True)
    recs, by = [], {}
    for k in sorted(set(it["k"] for it in items)):
        sub = [it for it in items if it["k"] == k]
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, n=k, seed=0)
        outs = llm.generate([it["prompt"] for it in sub], sp)
        for it, o in zip(sub, outs):
            oks, lens, trunc = [], [], []
            for c in o.outputs:
                g = c.text; ans = g.split("</think>")[-1] if "</think>" in g else g
                oks.append(check_answer(ans, "\\boxed{%s}" % it["gold"]) or check_answer(ans, it["gold"]))
                lens.append(len(c.token_ids)); trunc.append(c.finish_reason == "length")
            recs.append({"bench": it["bench"], "gold": it["gold"], "avg": sum(oks) / len(oks), "oks": oks,
                         "mean_len": sum(lens) / len(lens), "trunc": sum(trunc) / len(trunc)})
    for b in sorted(set(r["bench"] for r in recs)):
        rs = [r for r in recs if r["bench"] == b]
        by[b] = {"acc": round(sum(r["avg"] for r in rs) / len(rs), 4), "n": len(rs),
                 "mean_len": round(sum(r["mean_len"] for r in rs) / len(rs)), "trunc": round(sum(r["trunc"] for r in rs) / len(rs), 3)}
    aime = [r for r in recs if r["bench"].startswith("AIME")]
    res = {"model": args.model, "tag": args.tag, "k": args.k, "max_tokens": args.max_tokens, "think": not args.no_think,
           "aime_avg": round(sum(r["avg"] for r in aime) / len(aime), 4), "by_bench": by, "seconds": round(time.time() - t0), "records": recs}
    json.dump(res, open(f"{OUT}/eval_{args.tag}.json", "w"), indent=1)
    print(f"EVAL DONE {args.tag}: AIME avg@{args.k}={res['aime_avg']} | " + " ".join(f"{b}={v['acc']} (len {v['mean_len']}, trunc {v['trunc']})" for b, v in by.items()) + f" | {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
