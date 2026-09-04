"""Stream a per-step checkpoint series through the readout with bounded disk: download -> readout vs Base
(and vs the previous kept step) -> delete weights. Keeps a small set of steps on disk (--keep).

Usage: CUDA_VISIBLE_DEVICES=g python stream_steps.py SERIES STEPS [--keep 0,27]
  SERIES: DAPO | PPO | Spiral  (HF user caiyuchen, repos <SERIES>-step-<n>, base Qwen3-8B-Base)
  STEPS : comma list, e.g. 0,1,2,4,7,10,13,17,20,24,27
Markers: zoo/logs/ro_<series>_s<n>.DONE  (readout vs Base) and ro_<series>_s<prev>to<n>.DONE (vs previous step)
"""
import argparse, os, sys, shutil, subprocess, sys, time, glob
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/localscratch/zjin350/hf_cache"))
from huggingface_hub import snapshot_download
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); LOG = f"{ZOO}/logs"
PY = os.environ.get("ZOO_PY", sys.executable)
ap = argparse.ArgumentParser(); ap.add_argument("series"); ap.add_argument("steps"); ap.add_argument("--keep", default="")
ap.add_argument("--base", default="Qwen/Qwen3-8B-Base"); args = ap.parse_args()
steps = [int(s) for s in args.steps.split(",")]; keep = {int(s) for s in args.keep.split(",") if s}
ser = args.series.lower()


def log(m):
    line = f"{time.strftime('%F %T')} [{ser}] {m}"; print(line, flush=True); open(f"{LOG}/stream.log", "a").write(line + "\n")


def wait_gpu(min_free_gb=60):
    """GPUs are shared with other jobs: block until CUDA_VISIBLE_DEVICES has enough free memory for two 8B models."""
    gid = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    while True:
        try:
            free = float(subprocess.check_output(["nvidia-smi", "-i", gid, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode()) / 1024
        except Exception:
            free = 0
        if free >= min_free_gb: return
        log(f"waiting for GPU{gid} memory ({free:.0f}G free < {min_free_gb}G)"); time.sleep(120)


def run(cmd, logf):
    with open(logf, "w") as f:
        return subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ZOO)


def readout(a, b, tag):
    if os.path.exists(f"{LOG}/ro_{tag}.DONE"): return True
    wait_gpu()
    rc = run(f"{PY} {ZOO}/scripts/readout_pair.py '{a}' '{b}' {tag}", f"{LOG}/ro_{tag}.log")
    ok = rc == 0 and os.path.exists(f"{ZOO}/results/ro_{tag}_summary.json")
    open(f"{LOG}/ro_{tag}.{'DONE' if ok else 'FAIL'}", "w").write(f"rc={rc}\n"); log(f"{'DONE' if ok else 'FAIL'} {tag}")
    return ok


def repo_dir(n): return f"{os.environ['HF_HOME']}/hub/models--caiyuchen--{args.series}-step-{n}"


prev = None
for n in steps:
    repo = f"caiyuchen/{args.series}-step-{n}"
    while shutil.disk_usage(ZOO).free / 2**30 < 25:
        log(f"waiting for disk"); time.sleep(300)
    for attempt in range(3):
        try:
            p = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"], max_workers=8); break
        except Exception as e:
            log(f"download error {repo}: {str(e)[:120]}"); time.sleep(60); p = None
    if p is None: continue
    log(f"downloaded {repo}")
    readout(args.base, p, f"{ser}_s{n}")
    if prev is not None and prev[1] is not None:
        readout(prev[1], p, f"{ser}_s{prev[0]}to{n}")
    # delete previous step's weights unless kept
    if prev is not None and prev[0] not in keep and prev[1] is not None and os.path.isdir(repo_dir(prev[0])):
        shutil.rmtree(repo_dir(prev[0]), ignore_errors=True); log(f"deleted weights step {prev[0]}")
    prev = (n, p)
if prev is not None and prev[0] not in keep:
    shutil.rmtree(repo_dir(prev[0]), ignore_errors=True); log(f"deleted weights step {prev[0]}")
log("series done")
