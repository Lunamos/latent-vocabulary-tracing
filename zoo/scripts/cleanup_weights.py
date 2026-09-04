"""Delete a model's weights once every job that needs them is DONE:
  ro_<tag>.DONE (batch-1 readout), ro_itlens_<tag>.DONE if listed in jobs_itlens.txt, eval_<tag>.DONE if listed (uncommented) in eval_list.txt.
Never deletes Qwen3-8B-Base / Qwen3-8B. Loop; safe to restart. Log: zoo/logs/cleanup.log"""
import os, sys, time, shutil, glob
ZOO = "/localscratch/zjin350/Documents/jlen/zoo"; LOG = f"{ZOO}/logs"
sys.path.insert(0, f"{ZOO}/scripts"); sys.argv = ["x"]
from run_queue import resolve
KEEP_SPECS = {"Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"}
def log(m):
    line = f"{time.strftime('%F %T')} {m}"; print(line, flush=True); open(f"{LOG}/cleanup.log", "a").write(line + "\n")
def parse(path):
    out = {}
    for raw in open(path):
        s = raw.split("#")[0].strip()
        if not s: continue
        parts = [x.strip() for x in s.split("|")]
        out[parts[0]] = parts[1:3] if len(parts) > 2 else parts[1:2]
    return out
while True:
    b1 = parse(f"{ZOO}/data/jobs_batch1.txt"); itl = parse(f"{ZOO}/data/jobs_itlens.txt"); ev = parse(f"{ZOO}/data/eval_list.txt")
    base_tags = {t: ab[1] for t, ab in b1.items() if ab[0] == "Qwen/Qwen3-8B-Base" and ab[1] not in KEEP_SPECS}
    deleted = 0
    for tag, spec in base_tags.items():
        req = [f"{LOG}/ro_{tag}.DONE"]
        # every batch-1 job that uses this spec as A or B must be done
        for t2, ab in b1.items():
            if spec in ab: req.append(f"{LOG}/ro_{t2}.DONE")
        if f"itlens_{tag}" in itl: req.append(f"{LOG}/ro_itlens_{tag}.DONE")
        if tag in ev: req.append(f"{LOG}/eval_{tag}.DONE")
        if not all(os.path.exists(r) for r in req): continue
        p = resolve(spec)
        if p is None or not os.path.isdir(p): continue
        target = p if "::" in spec else os.path.dirname(os.path.dirname(p))
        sz = sum(os.path.getsize(f) for f in glob.glob(f"{target}/**/*", recursive=True) if os.path.isfile(f)) / 2**30
        shutil.rmtree(target, ignore_errors=True); deleted += 1
        log(f"deleted {tag} ({sz:.1f}G): {target}; free={shutil.disk_usage('/localscratch').free/2**30:.0f}G")
    remaining = [t for t, s in base_tags.items() if resolve(s) is not None and os.path.isdir(resolve(s) or "/nonexistent")]
    if not remaining:
        log("nothing left to delete; exiting"); break
    time.sleep(120)
