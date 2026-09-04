#!/bin/bash -l
# Build the jlen venv on ICE scratch (run on the login node with nohup; ~10-15 min)
set -e
module load python/3.12.5
cd ~/scratch/jlen
python -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers>=4.57" datasets accelerate huggingface_hub math-verify vllm ninja
pip install -q -e jacobian-lens
python - <<'PY'
import torch, transformers, vllm, jlens, math_verify
print("OK torch", torch.__version__, torch.version.cuda, "transformers", transformers.__version__, "vllm", vllm.__version__)
PY
echo ENV-SETUP-DONE
