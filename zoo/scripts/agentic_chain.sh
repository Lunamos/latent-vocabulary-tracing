#!/bin/bash
# after the eval runners exhaust the list, read the agentic panel on GPU0/GPU1 (stream_pairs waits for >=60G free per GPU)
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until [ $(grep -c "eval list exhausted" logs/evals.log) -ge 5 ]; do sleep 120; done
CUDA_VISIBLE_DEVICES=0 $PY scripts/stream_pairs.py data/jobs_agentic_a.txt --keep Qwen/Qwen3-8B-Base,Qwen/Qwen3-8B > logs/stream_agentic_a.out 2>&1 &
sleep 60
CUDA_VISIBLE_DEVICES=1 $PY scripts/stream_pairs.py data/jobs_agentic_b.txt --keep Qwen/Qwen3-8B-Base,Qwen/Qwen3-8B > logs/stream_agentic_b.out 2>&1 &
wait
echo "$(date +%T) agentic panel done" >> logs/after_queue.log
