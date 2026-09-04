"""Build the frozen probe set for the zoo readout and save it to zoo/data/probes.jsonl.

Kinds (30 each unless overridden):
  math    : MATH-lighteval test L3-5, seed 777 (IDENTICAL to opd/scripts/diff_lens.py build_probes),
            chat-rendered (SYSTEM_MATH, enable_thinking=False) + gold solution.
  code    : HumanEval, first 30 tasks; chat prompt "Complete the function" + prompt, response = canonical solution.
  agent   : OpenThoughts-Agent-v1-SFT terminal trajectories (first 30 with >=3 turns), chat-rendered,
            truncated to MAX_LEN tokens. prompt_len = tokens up to the end of the first user turn.
  neutral : wikitext[1500:1530] via jlens.examples.load_wikitext_prompts (IDENTICAL to diff_lens).

Text is tokenized by the Qwen3-8B-Base tokenizer (shared by every panel model); all readout scripts
must tokenize the stored `text` with truncation at MAX_LEN=640 and assert identical ids across models.
"""
import json, os, sys, hashlib
os.environ.setdefault("HF_HOME", "/localscratch/zjin350/hf_cache")
sys.path.insert(0, "/localscratch/zjin350/Documents/jlen/opd/scripts")
sys.path.insert(0, "/localscratch/zjin350/Documents/jlen/repro/scripts")
import common  # noqa: F401  (adds jacobian-lens to sys.path)
from opd_common import math_prompts, SYSTEM_MATH
from jlens.examples import load_wikitext_prompts
import transformers

OUT = os.environ.get("PROBES_OUT", "/localscratch/zjin350/Documents/jlen/zoo/data/probes.jsonl")
MAX_LEN = 640
N = int(os.environ.get("N_PER_KIND", 30))
SEED = int(os.environ.get("PROBE_SEED", 777)); WOFF = int(os.environ.get("WIKI_OFFSET", 1500)); HE_OFF = int(os.environ.get("HE_OFFSET", 0)); AG_SKIP = int(os.environ.get("AGENT_SKIP", 0))
# PROBE_TOK: tokenizer used for token counts (the reference/base model); TEMPLATE_TOK: tokenizer whose chat template renders
# the prompts (defaults to PROBE_TOK; needed when the base tokenizer ships no template, e.g. OLMo-3 base).
tok = transformers.AutoTokenizer.from_pretrained(os.environ.get("PROBE_TOK", "Qwen/Qwen3-8B-Base"))
ttok = transformers.AutoTokenizer.from_pretrained(os.environ.get("TEMPLATE_TOK", os.environ.get("PROBE_TOK", "Qwen/Qwen3-8B-Base")))


def chat(msgs, add_gen=True):
    try:
        return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen, enable_thinking=False)
    except TypeError:
        return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False))


probes = []
# math (verbatim protocol)
for r in math_prompts(split="test", n=N, seed=SEED):
    prefix = chat([{"role": "system", "content": SYSTEM_MATH}, {"role": "user", "content": r["problem"]}])
    probes.append({"kind": "math", "text": prefix + r["solution"], "prompt_len": ntok(prefix),
                   "meta": {"level": r["level"], "type": r["type"]}})

# code
from datasets import load_dataset
he = load_dataset("openai/openai_humaneval", split="test")
for i in range(HE_OFF, HE_OFF + N):
    r = he[i]
    user = ("Complete the following Python function. Return the full function in a ```python code block.\n\n"
            + r["prompt"].rstrip())
    prefix = chat([{"role": "user", "content": user}])
    resp = "```python\n" + r["prompt"].rstrip("\n") + "\n" + r["canonical_solution"].rstrip() + "\n```"
    probes.append({"kind": "code", "text": prefix + resp, "prompt_len": ntok(prefix),
                   "meta": {"task_id": r["task_id"]}})

# agent: mid-trajectory windows (the Terminus task header is ~1k tokens, too long for MAX_LEN=640),
# so each probe = short system prompt + one terminal-output user turn (tail-truncated) + the assistant JSON reply.
AGENT_SYS = ("You are an AI assistant solving command-line tasks in a Linux environment. "
             "Given the terminal output, respond with JSON containing \"analysis\", \"plan\", "
             "\"commands\" (a list of {\"keystrokes\", \"duration\"}) and \"task_complete\".")
ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train", streaming=True)
got = 0
for r in ds.skip(AG_SKIP).take(400):
    conv = r["conversations"]
    msgs = [{"role": m.get("role") or m.get("from"), "content": m.get("content") or m.get("value")} for m in conv]
    if any(not isinstance(m["content"], str) for m in msgs):
        continue
    # pick the second user->assistant exchange (index 2,3) when present, else the first
    k = 2 if len(msgs) >= 4 and msgs[2]["role"] == "user" and msgs[3]["role"] == "assistant" else None
    if k is None:
        continue
    user_ids = tok.encode(msgs[k]["content"], add_special_tokens=False)
    user_txt = tok.decode(user_ids[-300:])
    asst = msgs[k + 1]["content"].strip()
    if len(tok.encode(asst, add_special_tokens=False)) < 60:
        continue
    prefix = chat([{"role": "system", "content": AGENT_SYS}, {"role": "user", "content": user_txt}])
    probes.append({"kind": "agent", "text": prefix + asst, "prompt_len": ntok(prefix),
                   "meta": {"task": r.get("task"), "run_id": r.get("run_id"), "turn": k, "n_turns": len(msgs)}})
    got += 1
    if got >= N:
        break

# neutral (verbatim protocol)
for i, p in enumerate(load_wikitext_prompts(n_prompts=WOFF + N)[WOFF:WOFF + N]):
    probes.append({"kind": "neutral", "text": p, "prompt_len": 0, "meta": {"idx": WOFF + i}})

with open(OUT, "w") as f:
    for i, p in enumerate(probes):
        p["key"] = f"probe{i:03d}"
        p["n_tok"] = min(ntok(p["text"]), MAX_LEN)
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:12]
from collections import Counter
print(Counter(p["kind"] for p in probes), "sha256", h)
print("mean n_tok by kind:", {k: sum(p["n_tok"] for p in probes if p["kind"] == k) / N for k in ("math", "code", "agent", "neutral")})
print("mean prompt_len by kind:", {k: sum(p["prompt_len"] for p in probes if p["kind"] == k) / N for k in ("math", "code", "agent", "neutral")})
