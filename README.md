# Latent Vocabulary Tracing

Latent Vocabulary Tracing (LVT) is a research toolkit for comparing a language
model with a post-trained descendant in vocabulary space. It runs both models
on identical token sequences, decodes corresponding hidden states, and measures
how the decoded token distribution changes across layers and sequence roles.

The primary scalar is the **vocabulary write amount**

```text
W(parent → descendant) = KL(p_descendant || p_parent)
```

measured in nats after both distributions have been expressed in the same,
normally parent-anchored, vocabulary coordinate system. The direction is
intentional: it weights the log probability ratio by probability assigned by
the descendant. Reverse KL and Jensen–Shannon divergence are retained as
robustness statistics.

Beyond a single scalar, an LVT trace records:

1. write amount on task and unrelated inputs;
2. location across model depth and token roles;
3. category turnover, promotion/suppression balance, and turnover enrichment
   relative to each category's midpoint probability mass;
4. signed log-probability changes for concrete tokens; and
5. alignment between the token-change directions of two training runs.

Turnover enrichment is reported in bits. A value of `+1` means that a token
class carries twice the share of probability movement expected from the
class's mean parent/descendant probability mass. Raw turnover shares remain
part of the result so enrichment cannot hide a negligible edit.

The confirmatory taxonomy is deliberately conservative. Ambiguous formal
words such as `function`, `set`, and `vector` are not forced into mathematics
or programming, and ordinary English words that also happen to be language
keywords—such as `for`, `with`, `from`, `this`, and `new`—remain general
language. This prevents the vocabulary definition itself from manufacturing a
code signature.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,dev]'
```

The core package requires only NumPy. Model tracing and exploratory analysis
use the optional research dependencies.

## Command line

```bash
# Validate an experiment manifest before launching expensive jobs.
lvt manifest check zoo/data/jobs_frpo.txt

# Freeze both checkpoint direction and public lineage metadata.
lvt manifest check zoo/data/jobs_confirmatory_qwen8.txt \
  --edge-registry zoo/data/edge_registry.json

# Inspect a compact trace summary.
lvt summary zoo/results/ro_example_summary.json

# Refuse legacy/native results when building a confirmatory LL analysis.
lvt summary zoo/results/ro_example_summary.json --contract-readout LL \
  --require-categories --require-fp32-store \
  --edge-registry zoo/data/edge_registry.json

# Apply the built-in vocabulary taxonomy to literal tokenizer pieces.
lvt token classify ' therefore' '\boxed' '<|assistant|>'

# Use the paper's coarser functional classes.
lvt token classify --scheme functional ' therefore' '\boxed' ' equation'

# Use the confirmatory cross-domain classes (math, code, and agent/tool traces).
lvt token classify --scheme trace ' therefore' '\frac' ' stderr' 'def'
```

## Python API

```python
from latent_vocabulary_tracing import vocabulary_write_amount

# Last dimension is vocabulary; output retains the leading dimensions.
write_nats = vocabulary_write_amount(parent_logits, descendant_logits)
```

`vocabulary_write_amount` assumes its inputs already share one decoder-defined
coordinate system. It does not by itself anchor hidden states or validate that
two tokenizers are compatible.

## Repository map

- `src/latent_vocabulary_tracing/`: path-independent metrics, manifests,
  lineage registries, summary readers, taxonomy, and CLI.
- `tests/`: fast CPU tests for the stable package.
- `zoo/`: research runners and experiment manifests. This layer is still being
  migrated away from workstation-specific paths and is not yet a stable API.

Model weights, fitted lenses, tensor stores, logs, generated analyses, external
checkouts, manuscripts, and working documentation are excluded from version
control.

## Status

This is a research preview. The lightweight metrics and file-validation layer
is usable; the GPU runner remains experimental. In particular, legacy stores
may use a model-native decoder, while the primary LVT estimand requires the
parent decoder for both states. New analyses should record decoder mode
explicitly and should not compare the two quantities as if they were identical.

Copyright belongs to Lunamos. No public software license has been selected.
