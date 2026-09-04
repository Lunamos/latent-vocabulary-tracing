#!/bin/bash
# second pass over the step series: retries steps that failed (OOM from shared GPUs); skips DONE ones
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until grep -q "GPU2 chain done" logs/after_queue.log && grep -q "GPU3 lens chain done" logs/after_queue.log; do sleep 180; done
rm -f logs/ro_dapo_*.FAIL logs/ro_ppo_*.FAIL logs/ro_spiral_*.FAIL
CUDA_VISIBLE_DEVICES=3 $PY scripts/stream_steps.py DAPO 0,1,2,4,7,10,16,20,24 --keep 0 >> logs/stream_dapo.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/stream_steps.py PPO 0,2,5,9,14,19,23 >> logs/stream_ppo.out 2>&1
CUDA_VISIBLE_DEVICES=2 $PY scripts/stream_steps.py Spiral 0,2,5,9,14,18,22 >> logs/stream_spiral.out 2>&1
wait
echo "$(date +%T) step-series pass 2 done" >> logs/after_queue.log
