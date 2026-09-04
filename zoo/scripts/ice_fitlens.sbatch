#!/bin/bash -l
#SBATCH -A cse -q coe-ice -p coe-gpu --gres=gpu:h100:1 -c 8 --mem=96G -t 10:00:00
#SBATCH -J jlen-fitlens
#SBATCH --exclude=atl1-1-03-010-30-0
#SBATCH -o /storage/ice1/0/1/zjin350/jlen/jobs/logs/%x-%j.out
# Usage: sbatch ice_fitlens.sbatch MODEL START END OUT_PATH
module load python/3.12.5
cd ~/scratch/jlen
source .venv/bin/activate
export HF_HOME=$HOME/scratch/jlen/hf_cache
python repro/scripts/fit_lens.py "$@"
echo FITLENS-JOB-DONE
