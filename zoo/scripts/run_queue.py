"""GPU queue for zoo readouts.

Reads a jobs file: one job per line  `TAG | MODEL_A | MODEL_B [| extra args]`  ('#' comments).
MODEL_* may be an HF repo id (resolved via the download marker zoo/logs/dl_<slug>.DONE, or the HF cache),
optionally with '::subdir'. A job waits until both models are downloaded, then runs
readout_pair.py on a free GPU (from --gpus). Markers: zoo/logs/ro_<TAG>.DONE / .FAIL. Re-runnable.
"""
import argparse, os, sys, subprocess, sys, time, glob
ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = f"{ZOO}/logs"
PY = os.environ.get("ZOO_PY", sys.executable)
ap = argparse.ArgumentParser()
ap.add_argument("jobs"); ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--poll", type=int, default=30)
args = ap.parse_args() if __name__ == "__main__" else ap.parse_args(["/dev/null"])
os.environ["HF_HOME"] = "/localscratch/zjin350/hf_cache"


def log(m):
    line = f"{time.strftime('%F %T')} {m}"; print(line, flush=True)
    open(f"{LOG}/queue.log", "a").write(line + "\n")


def resolve(spec):
    """HF id or local path -> local path, or None if not yet downloaded."""
    spec = spec.strip()
    if os.path.isdir(spec):
        return spec
    repo, sub = (spec.split("::") + [None])[:2]
    marker = f"{LOG}/dl_{repo.replace('/', '__')}.DONE"
    if os.path.exists(marker):
        p = open(marker).read().strip()
    else:
        snaps = sorted(glob.glob(f"{os.environ['HF_HOME']}/hub/models--{repo.replace('/', '--')}/snapshots/*"))
        if not snaps:
            return None
        p = snaps[-1]
        if not glob.glob(f"{p}/**/*.safetensors", recursive=True):
            return None
    return os.path.join(p, sub) if sub else p


def load_jobs():
    jobs = []
    for raw in open(args.jobs):
        raw = raw.split("#")[0].strip()
        if not raw: continue
        parts = [x.strip() for x in raw.split("|")]
        tag, a, b = parts[:3]
        extra = parts[3] if len(parts) > 3 else ""
        jobs.append((tag, a, b, extra))
    return jobs


if __name__ == "__main__":
    gpus = [g for g in args.gpus.split(",") if g]
    running = {}  # gpu -> (tag, Popen)
    while True:
        # reap
        for g, (tag, pr) in list(running.items()):
            rc = pr.poll()
            if rc is None: continue
            del running[g]
            try: os.remove(f"{LOG}/ro_{tag}.CLAIM")
            except FileNotFoundError: pass
            if rc == 0 and os.path.exists(f"{ZOO}/results/ro_{tag}_summary.json"):
                open(f"{LOG}/ro_{tag}.DONE", "w").write("ok\n"); log(f"DONE {tag} on gpu{g}")
            else:
                open(f"{LOG}/ro_{tag}.FAIL", "w").write(f"rc={rc}\n"); log(f"FAIL {tag} rc={rc} on gpu{g}")
        jobs = load_jobs()
        pending = [j for j in jobs if not os.path.exists(f"{LOG}/ro_{j[0]}.DONE")
                   and not os.path.exists(f"{LOG}/ro_{j[0]}.FAIL") and not os.path.exists(f"{LOG}/ro_{j[0]}.CLAIM")
                   and j[0] not in [t for t, _ in running.values()]]
        if not pending and not running:
            log("queue empty, exiting"); break
        free = [g for g in gpus if g not in running]
        for g in free:
            for j in pending:
                tag, a, b, extra = j
                pa, pb = resolve(a), resolve(b)
                if pa is None or pb is None: continue
                cmd = f"CUDA_VISIBLE_DEVICES={g} {PY} {ZOO}/scripts/readout_pair.py '{pa}' '{pb}' {tag} {extra} > {LOG}/ro_{tag}.log 2>&1"
                open(f"{LOG}/ro_{tag}.CLAIM", "w").write(f"gpu{g} {time.strftime('%F %T')}\n")
                pr = subprocess.Popen(cmd, shell=True, cwd=ZOO, start_new_session=True)
                running[g] = (tag, pr); pending.remove(j); log(f"START {tag} on gpu{g}: {a} vs {b}")
                break
        time.sleep(args.poll)
