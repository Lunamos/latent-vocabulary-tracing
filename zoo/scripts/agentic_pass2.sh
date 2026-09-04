#!/bin/bash
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache HF_HUB_DISABLE_XET=1
PY=/localscratch/zjin350/Documents/jlen/jacobian-lens/.venv/bin/python
until grep -q "agentic panel done" logs/after_queue.log; do sleep 180; done
rm -f logs/ro_swelego.FAIL logs/ro_it_swelego.FAIL; sleep 30
CUDA_VISIBLE_DEVICES=0 $PY scripts/stream_pairs.py data/jobs_agentic.txt --keep Qwen/Qwen3-8B-Base,Qwen/Qwen3-8B > logs/stream_agentic_pass2.out 2>&1
echo "$(date +%T) agentic pass 2 done" >> logs/after_queue.log
