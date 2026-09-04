"""Sequential, disk-aware downloader for the Qwen3-8B-Base post-training zoo.
Usage: python download_zoo.py LISTFILE   (one repo id per line; '#' comments; optional 'repo::subdir' to fetch a subfolder)
Writes zoo/logs/download.log and touches zoo/logs/dl_<slug>.DONE per repo.
"""
import os, sys, shutil, time
os.environ.setdefault("HF_HOME", "/localscratch/zjin350/hf_cache")
from huggingface_hub import snapshot_download, HfApi
LOG = "/localscratch/zjin350/Documents/jlen/zoo/logs"
MIN_FREE_GB = 45
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py", "*.jinja", "*.tiktoken", "*.md"]
IGNORE = ["*.bin", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.ckpt", "*.onnx", "optimizer*", "*.pt"]
def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(f"{LOG}/download.log", "a") as f: f.write(line + "\n")
def free_gb(): return shutil.disk_usage("/localscratch").free / 2**30
api = HfApi()
for raw in open(sys.argv[1]):
    raw = raw.split("#")[0].strip()
    if not raw: continue
    repo, sub = (raw.split("::") + [None])[:2]
    slug = (repo + ("__" + sub if sub else "")).replace("/", "__")
    marker = f"{LOG}/dl_{slug}.DONE"
    if os.path.exists(marker): log(f"skip {raw} (done)"); continue
    while free_gb() < MIN_FREE_GB:
        log(f"WAIT free={free_gb():.0f}G < {MIN_FREE_GB}G before {raw}"); time.sleep(600)
    allow = [f"{sub}/*" for _ in [0]] if sub else ALLOW
    if sub: allow = [f"{sub}/{p}" for p in ALLOW]
    t0 = time.time()
    for attempt in range(3):
        try:
            p = snapshot_download(repo, allow_patterns=allow, ignore_patterns=IGNORE, max_workers=8)
            log(f"OK {raw} -> {p} ({time.time()-t0:.0f}s, free={free_gb():.0f}G)")
            open(marker, "w").write(p + "\n"); break
        except Exception as e:
            log(f"ERR {raw} attempt {attempt}: {type(e).__name__}: {str(e)[:300]}"); time.sleep(60)
log("ALL DONE")
