#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
CUDA_VISIBLE_DEVICES=3 $PY scripts/stream_steps.py DAPO 0,1,2,4,7,10,16,20,24 --keep 0 >> logs/stream_dapo.out 2>&1
echo "$(date +%T) GPU3 lens chain done" >> logs/after_queue.log
