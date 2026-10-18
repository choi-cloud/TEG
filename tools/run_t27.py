"""T27: TEG original (T22 `base` command) on ogbn-arxiv, 5-way {1, 5}-shot x seed 0-2 -> results/T27/teg/ogbn-arxiv_5w<k>s/seed<s>/.

Copied from tools/run_t22.py (config `base` only, dataset ogbn-arxiv). No code change to main.py / model.py / utils.py.
Additions (measurement only): stdout is read line by line into console.log (raw) and console_times.tsv (elapsed sec, line;
PYTHONUNBUFFERED=1); GPU memory.used of the run's GPU is polled every 1 s with nvidia-smi -> gpu_mem.json (max MiB).
Usage  python tools/run_t27.py phase1   -> 5-shot seed 0 alone on GPU 0
       python tools/run_t27.py phase2   -> remaining 5 runs on GPUs 0, 1, 2, 4, 5
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from run_t21 import GPUS  # noqa: E402

DS = "ogbn-arxiv"
RESULTS = os.path.join(ROOT, "results", "T27", "teg")
FIRST = (5, 0)
JOBS = [(k, s) for s in (0, 1, 2) for k in (1, 5)]
OUT_FILES = ("run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy", "split.json", "labels.npy")


def poll_mem(gpu, stop, box):
    """nvidia-smi is queried by physical index (a UUID query fails with GPU 3's "Unknown Error"); CUDA still uses the UUID."""
    while not stop.is_set():
        try:
            v = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
            box["max_mib"] = max(box["max_mib"], int(v.split()[0]))
            box["n"] += 1
        except Exception:
            box["err"] += 1
        stop.wait(1.0)


def run_once(gpu, k, s, log_name):
    rel = os.path.join("results", "T27", "teg", f"{DS}_5w{k}s", f"seed{s}")
    out = os.path.join(ROOT, rel)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "main.py", "--dataset", DS, "--way", "5", "--shot", str(k), "--seed", str(s), "--num_seed", "1",
           "--device", "0", "--pres_lambda", "0", "--pres_pool", "nb", "--dump_logits", "--dump_emb", "--fixed_eval", "--out_dir", rel]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu], PYTHONUNBUFFERED="1")
    stop, box = threading.Event(), {"max_mib": 0, "n": 0, "err": 0}
    th = threading.Thread(target=poll_mem, args=(gpu, stop, box), daemon=True)
    th.start()
    t0 = time.time()
    tag = "" if log_name == "console.log" else log_name[len("console"):-len(".log")]
    with open(os.path.join(out, log_name), "w") as log, open(os.path.join(out, f"console_times{tag}.tsv"), "w") as tl:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            tl.write(f"{time.time() - t0:.2f}\t{line.rstrip()}\n")
            tl.flush()
        rc = p.wait()
    t1 = time.time()
    stop.set()
    th.join()
    json.dump({"gpu": gpu, "uuid": GPUS[gpu], **box}, open(os.path.join(out, f"gpu_mem{tag}.json"), "w"))
    ok = rc == 0 and all(os.path.exists(os.path.join(out, f)) for f in OUT_FILES)
    return ok, rc, t0, t1, box["max_mib"]


def worker(gpu, q, lock):
    while True:
        try:
            k, s = q.get_nowait()
        except queue.Empty:
            return
        attempts = []
        for log_name in ("console.log", "console_retry1.log"):
            ok, rc, t0, t1, mem = run_once(gpu, k, s, log_name)
            attempts.append({"log": log_name, "exit": rc, "ok": ok, "start": t0, "end": t1, "sec": t1 - t0, "gpu_max_mib": mem})
            if ok:
                break
        rec = {"dataset": DS, "shot": k, "seed": s, "gpu": gpu, "ok": attempts[-1]["ok"], "attempts": attempts}
        with lock:
            with open(os.path.join(RESULTS, "_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {DS} 5w{k}s seed{s} ok={rec['ok']} tries={len(attempts)} "
                  f"sec={attempts[-1]['sec']:.1f} gpu_max_mib={attempts[-1]['gpu_max_mib']}", flush=True)


def main(jobs, gpus):
    os.makedirs(RESULTS, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(gpu, q, lock)) for gpu in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "phase1":
        main([FIRST], [0])
    elif sys.argv[1] == "phase2":
        main([j for j in JOBS if j != FIRST], [0, 1, 2, 4, 5])
