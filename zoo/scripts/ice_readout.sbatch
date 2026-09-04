#!/bin/bash -l
#SBATCH -A cse -q coe-ice -p coe-gpu --gres=gpu:h100:1 -c 8 --mem=96G -t 06:00:00
#SBATCH -J jlen-readout
#SBATCH -o /storage/ice1/0/1/zjin350/jlen/jobs/logs/%x-%j.out
# Usage: sbatch ice_readout.sbatch JOBSFILE [--keep repo,repo]
module load python/3.12.5
cd ~/scratch/jlen
source .venv/bin/activate
export HF_HOME=$HOME/scratch/jlen/hf_cache ZOO=$HOME/scratch/jlen/zoo ZOO_PY=$HOME/scratch/jlen/.venv/bin/python
export
python zoo/scripts/stream_pairs.py "$@" --min_free 50
echo READOUT-JOB-DONE
