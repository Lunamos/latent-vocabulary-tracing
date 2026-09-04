#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache HF_HUB_DISABLE_XET=1
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until grep -q "jobs exhausted" logs/stream_fll.out; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 $PY scripts/stream_pairs.py data/jobs_fll2.txt --keep Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-8B-Base >> logs/stream_fll2.out 2>&1
echo "$(date +%T) fll2 done" >> logs/after_queue.log
