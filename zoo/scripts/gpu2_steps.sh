#!/bin/bash
# DAPO + PPO per-step series on GPU2 (Spiral dropped: it is a 4B model, hidden 2560). Waits for the extra2 increments to finish first.
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache HF_HUB_DISABLE_XET=1
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until grep -q "extra2 increments done" logs/after_queue.log; do sleep 120; done
rm -f logs/ro_dapo_*.FAIL logs/ro_ppo_*.FAIL
CUDA_VISIBLE_DEVICES=2 $PY scripts/stream_steps.py DAPO 0,1,2,4,7,10,16,20,24 --keep 0 >> logs/stream_dapo.out 2>&1
CUDA_VISIBLE_DEVICES=2 $PY scripts/stream_steps.py PPO 0,2,5,9,14,19,23 >> logs/stream_ppo.out 2>&1
echo "$(date +%T) step series done on GPU2" >> logs/after_queue.log
