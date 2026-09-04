#!/usr/bin/env bash
# Wait for one handoff stream to settle, then run its CPU-only postprocessing.
# Usage: bash scripts/postprocess_after_streams.sh frpo|fll2
set -euo pipefail

ZOO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$ZOO_ROOT/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/jacobian-lens/.venv/bin/python"

queue_settled() {
  local job_file="$1"
  local pending=0
  local tag
  while IFS= read -r tag; do
    [ -z "$tag" ] && continue
    if [ ! -f "$ZOO_ROOT/logs/ro_${tag}.DONE" ] && [ ! -f "$ZOO_ROOT/logs/ro_${tag}.FAIL" ]; then
      pending=$((pending + 1))
    fi
  done < <(awk -F'|' '$0 !~ /^#/ && NF >= 3 {gsub(/^ +| +$/, "", $1); print $1}' "$job_file")
  [ "$pending" -eq 0 ]
}

wait_for_queue() {
  local job_file="$1"
  while ! queue_settled "$job_file"; do
    sleep 60
  done
}

postprocess_frpo() {
  wait_for_queue "$ZOO_ROOT/data/jobs_frpo.txt"
  mapfile -t tags < <(
    find "$ZOO_ROOT/results" -maxdepth 1 -name 'ro_frpo_*.pt' -printf '%f\n' |
      sed -e 's/^ro_//' -e 's/\.pt$//' | sort
  )
  if [ "${#tags[@]}" -lt 2 ]; then
    echo "FRPO postprocess skipped: fewer than two compatible stores"
    return 1
  fi
  cd "$ZOO_ROOT"
  ALIGNMENT_SUBDIR=frpo "$PYTHON_BIN" scripts/edit_alignment.py "${tags[@]}"
  echo "FRPO postprocess complete: ${#tags[@]} stores"
}

postprocess_fll2() {
  wait_for_queue "$ZOO_ROOT/data/jobs_fll2.txt"
  mapfile -t tags < <(
    awk -F'|' '$0 !~ /^#/ && NF >= 3 {gsub(/^ +| +$/, "", $1); print $1}' "$ZOO_ROOT/data/jobs_fll2.txt" |
      while IFS= read -r tag; do
        if [ -f "$ZOO_ROOT/results/ro_${tag}.pt" ] && [ -f "$ZOO_ROOT/logs/ro_${tag}.DONE" ]; then
          printf '%s\n' "$tag"
        fi
      done
  )
  if [ "${#tags[@]}" -eq 0 ]; then
    echo "FLL2 postprocess skipped: no completed stores"
    return 1
  fi
  cd "$PROJECT_ROOT"
  "$PYTHON_BIN" zoo/scripts/write_taxonomy.py "${tags[@]}" --readout LL --kind math
  "$PYTHON_BIN" zoo/scripts/write_taxonomy.py "${tags[@]}" --readout LL --kind neutral
  echo "FLL2 postprocess complete: ${#tags[@]} stores"
}

case "${1:-}" in
  frpo) postprocess_frpo ;;
  fll2) postprocess_fll2 ;;
  *) echo "usage: $0 frpo|fll2" >&2; exit 2 ;;
esac
