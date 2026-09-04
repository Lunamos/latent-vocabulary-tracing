#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
until grep -q "jobs exhausted" logs/stream_olmo.out; do sleep 60; done
rm -f logs/ro_olmo_*.FAIL
CUDA_VISIBLE_DEVICES=2 ../jacobian-lens/.venv/bin/python scripts/stream_pairs.py data/jobs_olmo.txt --keep allenai/Olmo-3-1025-7B > logs/stream_olmo_pass2.out 2>&1
echo "$(date +%T) OLMo pass-2 done" >> logs/after_queue.log
