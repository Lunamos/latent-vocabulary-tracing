#!/bin/bash
# re-run the pairs whose readout started before the 21:54:40 NaN patch
cd /localscratch/zjin350/Documents/jlen/zoo
for t in it klear_sft klear_rl casc_sft casc_rlhf casc_ifrl r1distill; do
  until [ -e logs/ro_$t.DONE ] || [ -e logs/ro_$t.FAIL ]; do sleep 10; done
  rm -f logs/ro_$t.DONE logs/ro_$t.FAIL results/ro_$t.pt results/ro_$t_summary.json
  echo "$(date +%T) requeued $t"
done
