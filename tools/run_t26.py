"""T26: train the official GPN on ogbn-arxiv with the T24 code path (tools/run_t24.py functions, unchanged).

Inputs: utils.load_data("ogbn-arxiv") -> adj (row-normalized D^-1(A+I), original edge direction), features (128), labels,
degree (row sums of A); classes from dataset/ogbn_arxiv/cls_split.pkl (fixed, seed-independent). Nodes with degree 0 get
degree 1 (log 1 = 0 -> support weight 0.5, as T24 degree-1 nodes); no other change to the formula.
Training / selection / offsets: run_t24.train_one with run_t24.load_inputs -> load_inputs_arxiv and run_t24.RESULTS ->
results/T26/gpn (episodes Random(18000 + s), valid pool Random(18100 + s), test pool Random(18200 + s), init 18300 + s).
Usage  python tools/run_t26.py hc                -> Jb: run_t24.hc_check re-run (output results/T26/hc_check.json)
       python tools/run_t26.py one --k <k> --seed <s>
       python tools/run_t26.py phase1            -> 5-shot seed 0 alone on GPU 0
       python tools/run_t26.py phase2            -> remaining 5 runs in parallel on GPUs 0, 1, 2, 4, 5
"""
import argparse
import hashlib
import json
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_t24 as g24  # noqa: E402  (first: sets the tools/gpn import path)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_t21 import GPUS  # noqa: E402

DS = "ogbn-arxiv"
RESULTS = os.path.join(ROOT, "results", "T26", "gpn")
SPLIT_PKL = os.path.join(ROOT, "dataset", "ogbn_arxiv", "cls_split.pkl")
FIRST = (5, 0)
JOBS = [(k, s) for s in (0, 1, 2) for k in (1, 5)]


def load_arxiv():
    from utils import load_data

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return load_data(DS)
    finally:
        os.chdir(cwd)


def fix_degree(degree):
    deg = degree.clone()
    zero = deg == 0
    deg[zero] = 1.0
    return deg, int(zero.sum())


def load_inputs_arxiv(ds, k, s):
    """Same return shape as run_t24.load_inputs."""
    _, adj, features, labels, degree, tr, va, te, _, _ = load_arxiv()
    deg, n0 = fix_degree(degree)
    sp = {"class_list_train": [int(c) for c in tr], "class_list_valid": [int(c) for c in va], "class_list_test": [int(c) for c in te]}
    y = labels.numpy()
    by_class = {c: np.where(y == c)[0].tolist() for g in sp for c in sp[g]}
    meta = {"split_source": os.path.relpath(SPLIT_PKL, ROOT), "split_sha256": hashlib.sha256(open(SPLIT_PKL, "rb").read()).hexdigest(),
            "labels_source": "utils.load_data", "degree0_set_to_1": n0}
    return adj, features, deg, labels.long(), sp, by_class, meta


def train_one(k, s):
    g24.load_inputs = load_inputs_arxiv
    g24.RESULTS = RESULTS
    torch.cuda.reset_peak_memory_stats()
    g24.train_one(DS, k, s)
    p = os.path.join(RESULTS, f"{DS}_5w{k}s", f"seed{s}", "run.json")
    r = json.load(open(p))
    r["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    r["gpu_max_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    json.dump(r, open(p, "w"), indent=1)


def hc():
    tmp = os.path.join(ROOT, "results", "T26", "_hc")
    orig = g24.ROOT
    g24.ROOT = tmp  # hc_check writes <ROOT>/results/T24/hc_check.json; keep the T24 file untouched
    try:
        g24.hc_check()
    finally:
        g24.ROOT = orig
    shutil.move(os.path.join(tmp, "results", "T24", "hc_check.json"), os.path.join(ROOT, "results", "T26", "hc_check.json"))
    shutil.rmtree(tmp)


# ---------------------------------------------------------------- driver (tools/run_t24.py pattern)
def run_once(gpu, k, s, log_name):
    out = os.path.join(RESULTS, f"{DS}_5w{k}s", f"seed{s}")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join("tools", "run_t26.py"), "one", "--k", str(k), "--seed", str(s)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu])
    t0 = time.time()
    with open(os.path.join(out, log_name), "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = rc == 0 and all(os.path.exists(os.path.join(out, f)) for f in ("run.json", "emb_best.npy", "score_best.npy", "ckpt_best.pt", "curves.json"))
    return ok, rc, t0, time.time()


def worker(gpu, q, lock):
    while True:
        try:
            k, s = q.get_nowait()
        except queue.Empty:
            return
        attempts = []
        for log_name in ("console.log", "console_retry1.log"):
            ok, rc, t0, t1 = run_once(gpu, k, s, log_name)
            attempts.append({"log": log_name, "exit": rc, "ok": ok, "start": t0, "end": t1, "sec": t1 - t0, "over_20min": t1 - t0 > 1200})
            if ok:
                break
        rec = {"dataset": DS, "shot": k, "seed": s, "gpu": gpu, "ok": attempts[-1]["ok"], "attempts": attempts}
        with lock:
            with open(os.path.join(RESULTS, "_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {DS} 5w{k}s seed{s} ok={rec['ok']} tries={len(attempts)} "
                  f"sec={attempts[-1]['sec']:.1f}", flush=True)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hc", "one", "phase1", "phase2"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.mode == "hc":
        hc()
    elif a.mode == "one":
        train_one(a.k, a.seed)
    elif a.mode == "phase1":
        main([FIRST], [0])
    else:
        main([j for j in JOBS if j != FIRST], [0, 1, 2, 4, 5])
