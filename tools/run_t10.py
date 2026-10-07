"""T10: configs {base, kl_h2 lambda 1, kl_h2 lambda 10} (pool nb) x 5-way {1,5}-shot;
original datasets (Amazon_clothing, dblp, Amazon_electronics) x seed 0-9 = 180 runs, plus datasets already in dataset/ and
readable by utils.load_data without code changes (corafull, coauthorCS, ogbn-arxiv) x seed 0-4 = 90 runs.

Copied from tools/run_t09.py; configs/datasets/seeds replaced. Order: seed -> config -> dataset -> shot. Cutoff 07:30 KST.

Seed-first queue (TN1 §4): seed 0's 6 settings -> seed 1's 6 -> ... ; within a seed the dataset
order follows T04 (Amazon_clothing -> dblp -> Amazon_electronics), 1-shot before 5-shot.
One run per GPU at a time; GPUs are selected by UUID (CLAUDE.md §4). A failed run is retried once
(console log `console_retry1.log`); if it fails again it is listed and the queue continues.
Per-run status is appended to results/T10/_runs.jsonl.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

GPUS = {
    0: "GPU-4b1a9f61-18d5-93a0-790e-87ba97de881e",
    1: "GPU-510575ba-3107-755b-c130-9eb114d07060",
    2: "GPU-2b6846d8-a8a0-c83b-dbf5-8bb183352d29",
    4: "GPU-e2a9ef79-7e90-3f82-2543-f8c2de140545",
    5: "GPU-b428f48d-97a3-a021-5db8-6abc37367a97",
    6: "GPU-ec7b3ebe-ce3c-ba3f-2b69-fdb715096a78",
}
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(10))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "T10")
CONFIGS = [("base", 0.0, []), ("kl_h2_l1", 1.0, []), ("kl_h2_l10", 10.0, [])]
EXTRA_DATASETS = ["corafull", "coauthorCS", "ogbn-arxiv"]
DEADLINE = 1791325800.0  # 2026-10-07 07:30 KST


def jobs():
    for s in SEEDS:
        for cfg in CONFIGS:
            for ds in DATASETS + (EXTRA_DATASETS if s < 5 else []):
                for k in SHOTS:
                    yield cfg, ds, k, s


def run_once(gpu, cfg, ds, k, s, log_name):
    name, lam, extra = cfg
    out_dir = os.path.join("results", "T10", name, f"{ds}_5w{k}s", f"seed{s}")
    os.makedirs(os.path.join(ROOT, out_dir), exist_ok=True)
    cmd = [sys.executable, "main.py", "--dataset", ds, "--way", "5", "--shot", str(k), "--seed", str(s),
           "--num_seed", "1", "--device", "0",
           "--pres_lambda", str(lam), "--pres_pool", "nb"] + extra + ["--dump_logits", "--dump_emb", "--fixed_eval", "--out_dir", out_dir]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu])
    t0 = time.time()
    with open(os.path.join(ROOT, out_dir, log_name), "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = rc == 0 and all(os.path.exists(os.path.join(ROOT, out_dir, f)) for f in ("run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy", "episodes.jsonl"))
    return ok, rc, t0, time.time()


def worker(gpu, q, lock):
    while True:
        try:
            cfg, ds, k, s = q.get_nowait()
        except queue.Empty:
            return
        attempts = []
        for log_name in ("console.log", "console_retry1.log"):
            if time.time() >= DEADLINE:
                attempts.append({"log": log_name, "exit": None, "ok": False, "not_started": "07:30 KST"})
                break
            ok, rc, t0, t1 = run_once(gpu, cfg, ds, k, s, log_name)
            attempts.append({"log": log_name, "exit": rc, "ok": ok, "start": t0, "end": t1, "sec": t1 - t0})
            if ok:
                break
        rec = {"config": cfg[0], "dataset": ds, "shot": k, "seed": s, "gpu": gpu, "ok": attempts[-1]["ok"], "attempts": attempts}
        with lock:
            with open(os.path.join(RESULTS, "_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {cfg[0]} {ds} 5w{k}s seed{s} ok={rec['ok']} "
                  f"tries={len(attempts)} sec={attempts[-1].get('sec', 0):.1f}", flush=True)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    q = queue.Queue()
    for j in jobs():
        q.put(j)
    lock = threading.Lock()
    t0 = time.time()
    threads = []
    for gpu in GPUS:
        th = threading.Thread(target=worker, args=(gpu, q, lock))
        th.start()
        threads.append(th)
        time.sleep(1)  # keep the queue order: seed-first dispatch
    for th in threads:
        th.join()
    print(f"total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
