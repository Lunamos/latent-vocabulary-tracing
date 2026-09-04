#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache HF_HUB_DISABLE_XET=1
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until grep -q "\[gpu0\] jobs exhausted" logs/stream_pairs.log; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 $PY scripts/stream_pairs.py data/jobs_q17_extra.txt --keep Qwen/Qwen3-1.7B-Base >> logs/stream_q17.out 2>&1
echo "$(date +%T) q17 extra done" >> logs/after_queue.log
