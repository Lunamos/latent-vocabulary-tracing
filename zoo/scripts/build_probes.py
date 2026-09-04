"""Build the frozen probe set for the zoo readout and save it to zoo/data/probes.jsonl.

Kinds (30 each unless overridden):
  math    : MATH-lighteval test L3-5, seed 777,
            chat-rendered (SYSTEM_MATH, enable_thinking=False) + gold solution.
  code    : HumanEval, first 30 tasks; chat prompt + canonical solution.
  agent   : OpenThoughts-Agent-v1-SFT terminal trajectories (first 30 with a second
            user/assistant exchange), chat-rendered and truncated to MAX_LEN tokens.
  neutral : 30 distinct WikiText articles selected at qualifying-record indices
            1500 + 97*i; the first 48 source tokens are chat context and the
            remainder is the teacher-forced response.

Text is tokenized by the Qwen3-8B-Base tokenizer (shared by every panel model); all
readout scripts must truncate at MAX_LEN=640 and assert identical IDs across models.
"""

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import transformers
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from latent_vocabulary_tracing.probes import select_spaced_text_records  # noqa: E402
from latent_vocabulary_tracing.spans import (  # noqa: E402
    infer_role_spans,
    validate_role_spans,
)

OUT = os.environ.get("PROBES_OUT", str(ROOT / "zoo" / "data" / "probes.jsonl"))
MAX_LEN = 640
N = int(os.environ.get("N_PER_KIND", 30))
SEED = int(os.environ.get("PROBE_SEED", 777))
WOFF = int(os.environ.get("WIKI_OFFSET", 1500))
HE_OFF = int(os.environ.get("HE_OFFSET", 0))
AG_SKIP = int(os.environ.get("AGENT_SKIP", 0))

# Freeze source snapshots rather than silently rebuilding a different panel
# when a dataset repository's main branch changes.
with (ROOT / "zoo" / "data" / "probe_sources.json").open() as provenance_file:
    SOURCE_PROVENANCE = json.load(provenance_file)["sources"]

SYSTEM_MATH = "Solve the math problem step by step. Put your final answer in \\boxed{}."
# PROBE_TOK supplies token counts; TEMPLATE_TOK supplies the rendered chat template.
# The latter defaults to PROBE_TOK, but can differ when a base ships no template.
tok = transformers.AutoTokenizer.from_pretrained(os.environ.get("PROBE_TOK", "Qwen/Qwen3-8B-Base"))
ttok = transformers.AutoTokenizer.from_pretrained(
    os.environ.get("TEMPLATE_TOK", os.environ.get("PROBE_TOK", "Qwen/Qwen3-8B-Base"))
)


def chat(msgs, add_gen=True):
    try:
        return ttok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_gen, enable_thinking=False
        )
    except TypeError:
        return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False))


probes = []
# Math. Draw extra deterministic candidates so an unusually long problem
# cannot consume the entire truncation window and leave no scored response.
math_source = SOURCE_PROVENANCE["math"]
math_rows = load_dataset(
    math_source["repository"],
    math_source["config"],
    split=math_source["split"],
    revision=math_source["revision"],
)
math_rows = math_rows.filter(lambda row: row["level"] in ("Level 3", "Level 4", "Level 5")).shuffle(
    seed=SEED
)
math_rows = math_rows.select(range(min(N + 50, len(math_rows))))
math_count = 0
for r in math_rows:
    prefix = chat(
        [{"role": "system", "content": SYSTEM_MATH}, {"role": "user", "content": r["problem"]}]
    )
    prompt_len = ntok(prefix)
    if prompt_len >= MAX_LEN:
        continue
    probes.append(
        {
            "kind": "math",
            "text": prefix + r["solution"],
            "prompt_len": prompt_len,
            "meta": {"level": r["level"], "type": r["type"]},
        }
    )
    math_count += 1
    if math_count == N:
        break
if math_count != N:
    raise RuntimeError("not enough mathematics probes with a response inside MAX_LEN")

# code
code_source = SOURCE_PROVENANCE["code"]
he = load_dataset(
    code_source["repository"],
    code_source["config"],
    split=code_source["split"],
    revision=code_source["revision"],
)
for i in range(HE_OFF, HE_OFF + N):
    r = he[i]
    user = (
        "Complete the following Python function. Return the full function "
        "in a ```python code block.\n\n" + r["prompt"].rstrip()
    )
    prefix = chat([{"role": "user", "content": user}])
    resp = (
        "```python\n" + r["prompt"].rstrip("\n") + "\n" + r["canonical_solution"].rstrip() + "\n```"
    )
    probes.append(
        {
            "kind": "code",
            "text": prefix + resp,
            "prompt_len": ntok(prefix),
            "meta": {"task_id": r["task_id"]},
        }
    )

# agent: mid-trajectory windows (the Terminus task header is ~1k tokens, too long for MAX_LEN=640),
# Each probe is a short system prompt, one tail-truncated terminal-output user turn,
# and the assistant JSON reply.
AGENT_SYS = (
    "You are an AI assistant solving command-line tasks in a Linux environment. "
    'Given the terminal output, respond with JSON containing "analysis", "plan", '
    '"commands" (a list of {"keystrokes", "duration"}) and "task_complete".'
)
agent_source = SOURCE_PROVENANCE["agent"]
ds = load_dataset(
    agent_source["repository"],
    agent_source["config"],
    split=agent_source["split"],
    streaming=True,
    revision=agent_source["revision"],
)
got = 0
for r in ds.skip(AG_SKIP).take(400):
    conv = r["conversations"]
    msgs = [
        {"role": m.get("role") or m.get("from"), "content": m.get("content") or m.get("value")}
        for m in conv
    ]
    if any(not isinstance(m["content"], str) for m in msgs):
        continue
    # pick the second user->assistant exchange (index 2,3) when present, else the first
    k = (
        2
        if len(msgs) >= 4 and msgs[2]["role"] == "user" and msgs[3]["role"] == "assistant"
        else None
    )
    if k is None:
        continue
    user_ids = tok.encode(msgs[k]["content"], add_special_tokens=False)
    user_txt = tok.decode(user_ids[-300:])
    asst = msgs[k + 1]["content"].strip()
    if len(tok.encode(asst, add_special_tokens=False)) < 60:
        continue
    prefix = chat([{"role": "system", "content": AGENT_SYS}, {"role": "user", "content": user_txt}])
    probes.append(
        {
            "kind": "agent",
            "text": prefix + asst,
            "prompt_len": ntok(prefix),
            "meta": {
                "task": r.get("task"),
                "run_id": r.get("run_id"),
                "turn": k,
                "n_turns": len(msgs),
            },
        }
    )
    got += 1
    if got >= N:
        break

# Neutral continuation. Raw Wikitext from the replication phase is not a
# suitable main baseline for chat-formatted response spans: it confounds domain
# with the presence of a user/assistant template. Preserve the source and
# index, but ask the model to continue from a fixed 48-token prefix.
NEUTRAL_SYSTEM = "Continue the supplied passage in neutral expository prose."
NEUTRAL_CONTEXT_TOKENS = 48
NEUTRAL_SOURCE_STRIDE = 97
neutral_sources = select_spaced_text_records(
    # The registry is also used here so the human-readable appendix and the
    # executable reconstruction cannot silently drift apart.
    load_dataset(
        SOURCE_PROVENANCE["neutral"]["repository"],
        SOURCE_PROVENANCE["neutral"]["config"],
        split=SOURCE_PROVENANCE["neutral"]["split"],
        revision=SOURCE_PROVENANCE["neutral"]["revision"],
    ),
    count=N,
    start=WOFF,
    stride=NEUTRAL_SOURCE_STRIDE,
)
for source in neutral_sources:
    source_ids = tok.encode(source["text"], add_special_tokens=False)
    context = tok.decode(source_ids[:NEUTRAL_CONTEXT_TOKENS])
    continuation = tok.decode(source_ids[NEUTRAL_CONTEXT_TOKENS:])
    prefix = chat(
        [
            {"role": "system", "content": NEUTRAL_SYSTEM},
            {
                "role": "user",
                "content": "Continue this encyclopedia passage:\n\n" + context,
            },
        ]
    )
    probes.append(
        {
            "kind": "neutral",
            "text": prefix + continuation,
            "prompt_len": ntok(prefix),
            "meta": {
                "idx": source["qualifying_index"],
                "control": "chat_matched_continuation",
                "source_context_tokens": NEUTRAL_CONTEXT_TOKENS,
                "article": source["article"],
                "source_dataset": "Salesforce/wikitext:wikitext-103-raw-v1",
                "source_split": "train",
                "source_sampling": "spaced_qualifying_records",
                "source_start": WOFF,
                "source_stride": NEUTRAL_SOURCE_STRIDE,
            },
        }
    )

with open(OUT, "w") as f:
    for i, p in enumerate(probes):
        p["key"] = f"probe{i:03d}"
        p["n_tok"] = min(ntok(p["text"]), MAX_LEN)
        encoded = tok(
            p["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        role_spans = infer_role_spans(
            p["text"], kind=p["kind"], prompt_len=p["prompt_len"], offsets=offsets
        )
        validate_role_spans(role_spans, n_tokens=len(offsets))
        p["role_spans"] = {
            name: [list(span) for span in spans] for name, spans in role_spans.items()
        }
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:12]
print(Counter(p["kind"] for p in probes), "sha256", h)
print(
    "mean n_tok by kind:",
    {
        k: sum(p["n_tok"] for p in probes if p["kind"] == k) / N
        for k in ("math", "code", "agent", "neutral")
    },
)
print(
    "mean prompt_len by kind:",
    {
        k: sum(p["prompt_len"] for p in probes if p["kind"] == k) / N
        for k in ("math", "code", "agent", "neutral")
    },
)
