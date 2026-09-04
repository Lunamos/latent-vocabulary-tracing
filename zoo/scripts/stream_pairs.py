"""Disk-bounded pair readouts: for each job line `TAG | A | B | extra`, download A and B (spec = repo[@revision][::subdir]
or local path), run readout_pair.py, then delete B's weights unless its repo is listed in --keep (A is never deleted here).
Usage: CUDA_VISIBLE_DEVICES=g python stream_pairs.py JOBS [--keep repo1,repo2] [--min_free 25]
Markers: zoo/logs/ro_<TAG>.DONE/.FAIL (skip if present); zoo/logs/stream_pairs.log
"""
import argparse, os, sys, shutil, subprocess, sys, time, glob
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/localscratch/zjin350/hf_cache"))
from huggingface_hub import snapshot_download
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); LOG = f"{ZOO}/logs"; HF = os.environ["HF_HOME"] + "/hub"
PY = os.environ.get("ZOO_PY", sys.executable)
ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("--keep", default=""); ap.add_argument("--min_free", type=float, default=25)
args = ap.parse_args(); keep = set(x for x in args.keep.split(",") if x)
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model", "*.py", "*.tiktoken"]


def log(m):
    line = f"{time.strftime('%F %T')} [gpu{os.environ.get('CUDA_VISIBLE_DEVICES','?')}] {m}"; print(line, flush=True)
    open(f"{LOG}/stream_pairs.log", "a").write(line + "\n")


def try_claim(tag):
    """Atomically claim a tag so several GPU workers can share one queue."""
    path = f"{LOG}/ro_{tag}.CLAIM"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as f:
        f.write(f"pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} time={time.strftime('%F %T')}\n")
    return path


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


def sanitize_config(path):
    """Some released configs store integer fields as floats (e.g. max_position_embeddings: 163840.0); transformers 5 rejects them."""
    import json as _json
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg): return
    try:
        d = _json.load(open(cfg))
    except Exception:
        return
    changed = False
    def fix(o):
        nonlocal changed
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, float) and v.is_integer() and any(s in k for s in ("position", "size", "layers", "heads", "dim", "vocab", "token_id", "window", "length")):
                    o[k] = int(v); changed = True
                else: fix(v)
        elif isinstance(o, list):
            for v in o: fix(v)
    fix(d)
    if changed:
        os.chmod(cfg, 0o644); _json.dump(d, open(cfg, "w"), indent=2); log(f"sanitized integral floats in {cfg}")


def fetch(spec):
    if os.path.isdir(spec): return spec, None
    repo, sub = (spec.split("::") + [None])[:2]
    repo, rev = (repo.split("@") + [None])[:2]
    while shutil.disk_usage(ZOO).free / 2**30 < args.min_free:
        log(f"waiting for disk ({shutil.disk_usage(ZOO).free/2**30:.0f}G free)"); time.sleep(300)
    for attempt in range(3):
        try:
            allow = [f"{sub}/{p}" for p in ALLOW] + ["*.json", "*.txt", "*.jinja"] if sub else ALLOW
            p = snapshot_download(repo, revision=rev, allow_patterns=allow, max_workers=8)
            if sub and not os.path.exists(os.path.join(p, sub, "config.json")):
                for f in glob.glob(f"{p}/*.json") + glob.glob(f"{p}/*.txt") + glob.glob(f"{p}/*.jinja"):
                    if not os.path.exists(os.path.join(p, sub, os.path.basename(f))): shutil.copy(f, os.path.join(p, sub))
            sanitize_config(os.path.join(p, sub) if sub else p)
            return (os.path.join(p, sub) if sub else p), repo
        except Exception as e:
            log(f"download error {spec}: {type(e).__name__} {str(e)[:150]}"); time.sleep(60)
    return None, repo


def delete(path, repo):
    if repo is None or repo in keep: return
    # revision snapshot dir: delete just that snapshot (other revisions untouched); subfolder: delete subfolder
    target = path if "/snapshots/" in path else path
    sz = sum(os.path.getsize(f) for f in glob.glob(f"{target}/**/*", recursive=True) if os.path.isfile(f)) / 2**30
    shutil.rmtree(target, ignore_errors=True)
    # also drop the blobs that are now orphaned (revision snapshots symlink into blobs/)
    root = f"{HF}/models--{repo.replace('/', '--')}"
    linked = set()
    for f in glob.glob(f"{root}/snapshots/**/*", recursive=True):
        if os.path.islink(f): linked.add(os.path.realpath(f))
    for b in glob.glob(f"{root}/blobs/*"):
        if os.path.realpath(b) not in linked:
            try: os.remove(b)
            except OSError: pass
    log(f"deleted {target} ({sz:.1f}G); free={shutil.disk_usage(ZOO).free/2**30:.0f}G")


for raw in open(args.jobs):
    raw = raw.split("#")[0].strip()
    if not raw: continue
    parts = [x.strip() for x in raw.split("|")]; tag, a, b = parts[:3]; extra = parts[3] if len(parts) > 3 else ""
    if os.path.exists(f"{LOG}/ro_{tag}.DONE") or os.path.exists(f"{LOG}/ro_{tag}.FAIL"): continue
    # Wait before claiming: a worker on a busy GPU must not reserve work that an
    # idle GPU could execute. Recheck completion after the wait for long queues.
    wait_gpu()
    if os.path.exists(f"{LOG}/ro_{tag}.DONE") or os.path.exists(f"{LOG}/ro_{tag}.FAIL"): continue
    claim = try_claim(tag)
    if claim is None: continue
    try:
        pa, ra = fetch(a); pb, rb = fetch(b)
        if pa is None or pb is None:
            log(f"SKIP {tag}: download failed")
            continue
        # The GPU may have become busy while weights were downloading.
        wait_gpu()
        log(f"START {tag}: {a} vs {b}")
        rc = subprocess.call(f"{PY} {ZOO}/scripts/readout_pair.py '{pa}' '{pb}' {tag} {extra} > {LOG}/ro_{tag}.log 2>&1", shell=True, cwd=ZOO)
        ok = rc == 0 and os.path.exists(f"{ZOO}/results/ro_{tag}_summary.json")
        open(f"{LOG}/ro_{tag}.{'DONE' if ok else 'FAIL'}", "w").write(f"rc={rc}\n"); log(f"{'DONE' if ok else 'FAIL'} {tag} rc={rc}")
        delete(pb, rb)
    finally:
        try: os.remove(claim)
        except FileNotFoundError: pass
log("jobs exhausted")
