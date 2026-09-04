#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
until grep -q "GPU2 chain done" logs/after_queue.log; do sleep 120; done
CUDA_VISIBLE_DEVICES=2 ../jacobian-lens/.venv/bin/python scripts/stream_pairs.py data/jobs_extra2.txt > logs/stream_extra2.out 2>&1
echo "$(date +%T) extra2 increments done on GPU2" >> logs/after_queue.log
