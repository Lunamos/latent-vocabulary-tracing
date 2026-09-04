#!/bin/bash -l
#SBATCH -A cse -q coe-ice -p coe-gpu --gres=gpu:h100:1 -c 8 --mem=64G -t 08:00:00
#SBATCH -J jlen-eval32k
#SBATCH -o /storage/ice1/0/1/zjin350/jlen/jobs/logs/%x-%j.out
# Usage: sbatch ice_eval32k.sbatch LISTFILE   (list of "TAG | HF_ID" lines; runs sequentially, skips DONE)
module load python/3.12.5
cd ~/scratch/jlen
source .venv/bin/activate
export HF_HOME=$HOME/scratch/jlen/hf_cache
export PATH=$HOME/scratch/jlen/.venv/bin:$PATH
LIST=${1:-zoo/data/eval32k_list.txt}
while IFS= read -r raw; do
  line="${raw%%#*}"; [ -z "${line// }" ] && continue
  tag=$(echo "$line" | cut -d'|' -f1 | xargs); repo=$(echo "$line" | cut -d'|' -f2 | xargs)
  [ -e zoo/logs/eval32k_$tag.DONE ] && continue
  [ -e zoo/logs/eval32k_$tag.CLAIM ] && continue
  echo $SLURM_JOB_ID > zoo/logs/eval32k_$tag.CLAIM
  echo "$(date +%T) START $tag $repo"
  python zoo/scripts/eval_zoo.py "$repo" "${tag}_32k" --k 4 --max_tokens 30720 --max_model_len 32768 --n_math 100 > zoo/logs/eval32k_$tag.log 2>&1 \
    && touch zoo/logs/eval32k_$tag.DONE && echo "$(date +%T) DONE $tag" || echo "$(date +%T) FAIL $tag"
  rm -f zoo/logs/eval32k_$tag.CLAIM
  # free the weights (ICE scratch is plentiful, but keep the cache tidy): only this repo's snapshot dir
  rm -rf "$HF_HOME/hub/models--${repo//\//--}"
done < "$LIST"
echo ALL-DONE
