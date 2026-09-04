"""Sequential vLLM eval runner for one GPU (run several instances on different GPUs; CLAIM markers make it safe).
Usage: CUDA_VISIBLE_DEVICES=g python run_evals.py data/eval_list.txt
"""
import os, sys, subprocess, time, glob
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"; LOG = f"{ZOO}/logs"
PY = "/localscratch/zjin350/Documents/jlen/opd/.venv-vllm/bin/python"
os.environ["HF_HOME"] = "/localscratch/zjin350/hf_cache"
os.environ["PATH"] = "/localscratch/zjin350/Documents/jlen/opd/.venv-vllm/bin:" + os.environ.get("PATH", "")  # ninja etc.
sys.path.insert(0, f"{ZOO}/scripts")
from run_queue import resolve  # noqa
def log(m):
    line = f"{time.strftime('%F %T')} {m}"; print(line, flush=True); open(f"{LOG}/evals.log", "a").write(line + "\n")
for raw in open(sys.argv[1]):
    raw = raw.split("#")[0].strip()
    if not raw: continue
    parts = [x.strip() for x in raw.split("|")]; tag, spec = parts[0], parts[1]; extra = parts[2] if len(parts) > 2 else ""
    if any(os.path.exists(f"{LOG}/eval_{tag}.{s}") for s in ("DONE", "CLAIM", "FAIL")): continue
    p = resolve(spec)
    if p is None: log(f"skip {tag}: not downloaded"); continue
    open(f"{LOG}/eval_{tag}.CLAIM", "w").write(os.environ.get("CUDA_VISIBLE_DEVICES", "?") + "\n")
    log(f"START eval {tag} on gpu{os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    rc = subprocess.call(f"{PY} {ZOO}/scripts/eval_zoo.py '{p}' {tag} {extra} > {LOG}/eval_{tag}.log 2>&1", shell=True, cwd=ZOO)
    os.remove(f"{LOG}/eval_{tag}.CLAIM")
    ok = rc == 0 and os.path.exists(f"{ZOO}/results/eval_{tag}.json")
    open(f"{LOG}/eval_{tag}.{'DONE' if ok else 'FAIL'}", "w").write(f"rc={rc}\n"); log(f"{'DONE' if ok else 'FAIL'} eval {tag} rc={rc}")
log("eval list exhausted")
