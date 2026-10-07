"""T16a gate / dynamic-fact runs (all through tools/t16a_gate_run.py). Seed 0. GPU queue as tools/run_t14.py.
Usage: python tools/t16a_run_gates.py <old_worktree>"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

GPUS = {0: "GPU-4b1a9f61-18d5-93a0-790e-87ba97de881e", 1: "GPU-510575ba-3107-755b-c130-9eb114d07060",
        2: "GPU-2b6846d8-a8a0-c83b-dbf5-8bb183352d29", 4: "GPU-e2a9ef79-7e90-3f82-2543-f8c2de140545",
        5: "GPU-b428f48d-97a3-a021-5db8-6abc37367a97", 6: "GPU-ec7b3ebe-ce3c-ba3f-2b69-fdb715096a78"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.abspath(sys.argv[1])
RES = os.path.join(ROOT, "results", "T16a")
LAM = {("Amazon_clothing", 1): 10, ("Amazon_clothing", 5): 30, ("dblp", 1): 30, ("dblp", 5): 10,
       ("Amazon_electronics", 1): 100, ("Amazon_electronics", 5): 30}
GP = ["--grad_probe_steps", "1,10,50,250,500"]
DS = ["Amazon_clothing", "dblp", "Amazon_electronics"]


def infonce(ds, k):
    return ["--pres_lambda", str(LAM[(ds, k)]), "--pres_pool", "nb", "--pres_loss", "infonce", "--pres_m", "1024", "--neg_sampler", "uniform"]


JOBS = []  # (name, code_root, ds, k, extra)
for ds, k in (("Amazon_clothing", 1), ("dblp", 5)):
    JOBS += [(f"old/old1_{ds}_{k}", OLD, ds, k, []), (f"old/old2_{ds}_{k}", OLD, ds, k, []), (f"G1_new_{ds}_{k}", ROOT, ds, k, []),
             (f"G2_hooks_{ds}_{k}", ROOT, ds, k, ["--traj_every", "10", "--traj_lr", "1", "--traj_dump_ends", "1", "--grad_probe_steps", "1,50"])]
for ds in ("Amazon_clothing", "dblp"):
    JOBS.append((f"G4_relabel_{ds}_1", ROOT, ds, 1, ["--relabel_base", "1"]))
JOBS.append(("G4_default_dblp_1", ROOT, "dblp", 1, []))
for ds in DS:
    for k in (1, 5):
        JOBS.append((f"D1_{ds}_{k}", ROOT, ds, k, GP))
for ds in DS:
    JOBS += [(f"D2_g1_{ds}_1", ROOT, ds, 1, ["--gamma", "1.0"] + GP), (f"D2_g0_{ds}_1", ROOT, ds, 1, ["--gamma", "0.0"] + GP)]
JOBS.append(("D3_sup0_Amazon_clothing_1", ROOT, "Amazon_clothing", 1, ["--sup_coef", "0"] + GP))
for ds in DS:
    JOBS.append((f"D4_infonce_{ds}_1", ROOT, ds, 1, infonce(ds, 1) + GP))
JOBS += [("J4_dumpemb_Amazon_clothing_1", ROOT, "Amazon_clothing", 1, ["--dump_emb"]),
         ("J6_gcn128_Amazon_clothing_1", ROOT, "Amazon_clothing", 1, ["--gcn_out", "128"]),
         ("G8_traj10_Amazon_clothing_1", ROOT, "Amazon_clothing", 1, ["--traj_every", "10"]),
         ("G8_t17_Amazon_clothing_1", ROOT, "Amazon_clothing", 1, ["--sup_coef", "0", "--traj_every", "50", "--traj_lr", "1"] + infonce("Amazon_clothing", 1)),
         ("G8_t17_l2_Amazon_clothing_1", ROOT, "Amazon_clothing", 1, ["--sup_coef", "0", "--traj_every", "50", "--traj_lr", "1", "--gcn_layers", "2"] + infonce("Amazon_clothing", 1)),
         ("G8_traj10_Amazon_electronics_5", ROOT, "Amazon_electronics", 5, ["--traj_every", "10"]),
         ("G8_off_Amazon_electronics_5", ROOT, "Amazon_electronics", 5, [])]


def worker(gpu, q, lock):
    while True:
        try:
            name, code, ds, k, extra = q.get_nowait()
        except queue.Empty:
            return
        out = os.path.join(RES, name)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "tools", "t16a_gate_run.py"), code, out, "0", "--", "--dataset", ds, "--way", "5",
               "--shot", str(k), "--fixed_eval", "--dump_logits"] + extra
        t0 = time.time()
        with open(os.path.join(out, "console.log"), "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu]), stdout=log, stderr=subprocess.STDOUT)
        rec = {"name": name, "gpu": gpu, "exit": rc, "sec": time.time() - t0}
        with lock:
            with open(os.path.join(RES, "_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] {rec}", flush=True)


q = queue.Queue()
for j in JOBS:
    q.put(j)
lock = threading.Lock()
ths = [threading.Thread(target=worker, args=(g, q, lock)) for g in GPUS]
for t in ths:
    t.start()
    time.sleep(1)
for t in ths:
    t.join()
print("jobs", len(JOBS))
