#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
CUDA_VISIBLE_DEVICES=2 $PY scripts/stream_steps.py PPO 0,2,5,9,14,19,23 >> logs/stream_ppo.out 2>&1
CUDA_VISIBLE_DEVICES=2 $PY scripts/stream_steps.py Spiral 0,2,5,9,14,18,22 >> logs/stream_spiral.out 2>&1
echo "$(date +%T) GPU2 chain done" >> logs/after_queue.log
