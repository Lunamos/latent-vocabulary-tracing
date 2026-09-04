#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache HF_HUB_DISABLE_XET=1
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until grep -q "GPU3 lens chain done" logs/after_queue.log; do sleep 120; done
CUDA_VISIBLE_DEVICES=3 $PY scripts/stream_pairs.py data/jobs_batch1b.txt --keep Qwen/Qwen3-8B > logs/stream_b1b.out 2>&1
echo "$(date +%T) extra-Qwen chain done on GPU3" >> logs/after_queue.log
