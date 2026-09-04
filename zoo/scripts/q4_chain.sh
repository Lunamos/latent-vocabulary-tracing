#!/bin/bash
# wait for the 4B merged lens on ICE, fetch it, then stream the Qwen3-4B family on GPU1
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache HF_HUB_DISABLE_XET=1
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python; LENS=/localscratch/zjin350/Documents/jlen/repro/lenses/qwen3-4b_merged.pt
until [ -s "$LENS" ]; do
  if timeout 60 ssh zjin350@login-ice.pace.gatech.edu 'grep -q saved ~/scratch/jlen/jobs/logs/merge-4b.out'; then
    timeout 600 rsync -a zjin350@login-ice.pace.gatech.edu:'~/scratch/jlen/repro/lenses/qwen3-4b_merged.pt' /localscratch/zjin350/Documents/jlen/repro/lenses/
  else sleep 60; fi
done
echo "$(date +%T) 4B lens ready" >> logs/after_queue.log
CUDA_VISIBLE_DEVICES=1 $PY scripts/stream_pairs.py data/jobs_q4.txt --keep Qwen/Qwen3-4B-Base,Qwen/Qwen3-4B >> logs/stream_q4.out 2>&1
echo "$(date +%T) q4 stream done" >> logs/after_queue.log
