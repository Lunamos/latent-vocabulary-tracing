#!/bin/bash
# after both eval runners exhaust the list, clear FAIL markers (transient GPU-memory races) and run one more pass on GPU0/1
cd /localscratch/zjin350/Documents/jlen/zoo; export HF_HOME=/localscratch/zjin350/hf_cache
until [ $(grep -c "eval list exhausted" logs/evals.log) -ge 5 ]; do sleep 120; done
n=$(ls logs/eval_*.FAIL 2>/dev/null | wc -l); echo "$(date +%T) second pass: $n FAIL markers cleared" >> logs/after_queue.log
rm -f logs/eval_*.FAIL
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g setsid nohup ../jacobian-lens/.venv/bin/python scripts/run_evals.py data/eval_list.txt > logs/evals_pass2_g$g.out 2>&1 < /dev/null & sleep 90; done
