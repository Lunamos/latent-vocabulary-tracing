#!/bin/bash
# Waits for both batch-1 queue instances to report empty, then: GPUs 0-2 -> standardized evals; GPU3 -> IT-lens robustness readouts, then DAPO/PPO/Spiral step streaming.
cd /localscratch/zjin350/Documents/jlen/zoo
export HF_HOME=/localscratch/zjin350/hf_cache
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until [ $(grep -c "queue empty" logs/queue.log) -ge 2 ]; do sleep 60; done
echo "$(date +%T) batch-1 queue empty; launching evals on GPU0-2 and lens work on GPU3" >> logs/after_queue.log
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup $PY scripts/run_evals.py data/eval_list.txt > logs/evals_g$g.out 2>&1 < /dev/null &
  sleep 20
done
(
  $PY scripts/run_queue.py data/jobs_itlens.txt --gpus 3 > logs/queue_itlens_g3.out 2>&1
  CUDA_VISIBLE_DEVICES=3 $PY scripts/stream_steps.py DAPO 0,1,2,4,7,10,16,20,24 --keep 0 > logs/stream_dapo.out 2>&1
  CUDA_VISIBLE_DEVICES=3 $PY scripts/stream_steps.py PPO 0,2,5,9,14,19,23 > logs/stream_ppo.out 2>&1
  CUDA_VISIBLE_DEVICES=3 $PY scripts/stream_steps.py Spiral 0,2,5,9,14,18,22 > logs/stream_spiral.out 2>&1
  echo "$(date +%T) GPU3 lens chain done" >> logs/after_queue.log
) &
wait
echo "$(date +%T) after_queue all done" >> logs/after_queue.log
