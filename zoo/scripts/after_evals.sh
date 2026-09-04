#!/bin/bash
# When the eval list is exhausted and a GPU in {0,1,2} is idle, start the OLMo chain (GPU2) and the extra-Qwen chain (GPU1).
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 1500 ]; }
until grep -q "eval list exhausted" logs/evals.log; do sleep 60; done
until idle 2; do sleep 60; done
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY scripts/stream_pairs.py data/jobs_olmo.txt --keep allenai/Olmo-3-1025-7B > logs/stream_olmo.out 2>&1 < /dev/null &
echo "$(date +%T) OLMo chain started on GPU2" >> logs/after_queue.log
until idle 1; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY scripts/stream_pairs.py data/jobs_extra_qwen.txt --keep Qwen/Qwen3-8B-Base > logs/stream_extra.out 2>&1 < /dev/null &
echo "$(date +%T) extra-Qwen chain started on GPU1" >> logs/after_queue.log
