"""T29: published SGC baseline (Meta-GNN public-code procedure) in three variants on the T22 fixed test episodes.

Arms per (ds, seed), both shots, 200 fixed test episodes each (T22 `base` fixed_eval_logits.npz mode 1, ep_idx order):
  D      row-L2 Â²X (fusion_baseline.diffusion_features), cosine prototype
  SGC-P  V_sgc = Â²(row-L1 X), no post-normalization, cosine prototype
  SGC-S  linear layer, random init (torch.Generator 18800 + seed for 1-shot / 18900 + seed for 5-shot, ep_idx order) -> fit on support
  SGC-B  linear layer pretrained on base classes (ITER 50 x STEP 50, 20 nodes per class; random.Random(18600 + seed),
         torch.Generator(18700 + seed) init) -> fit on support
  LR     sklearn LogisticRegression(C=1.0, max_iter=1000) on row-L2 Â²X, support only (record)
init_linear / fit_linear / pretraining loop are ported from instructions/ref/pilot_sgc_vs_d.py (lines 82-97, 120-134).
Splits: restored per T28_T29_QUEUE §2.1 -> results/T29/splits.json (gate Gs). No global random / torch RNG use.
Usage  python tools/run_t29.py splits            -> results/T29/splits.json + Gs
       python tools/run_t29.py one <ds> <seed>    -> results/T29/<ds>/seed<s>/{run.json, per_episode.npz, pretrain.json}
       python tools/run_t29.py phase1             -> Amazon_clothing seed 0 alone on GPU 0 (Hc)
       python tools/run_t29.py phase2             -> remaining 29 processes on GPUs 0, 1, 2, 4, 5, 6
"""
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(10))
N_WAY = 5
ITER, STEP, TRAIN_SHOT = 50, 50, 20
VALID_NUM = {"Amazon_clothing": 17, "Amazon_electronics": 36, "dblp": 27}  # utils.py:13 valid_num_dic
R22B = os.path.join(ROOT, "results", "T22", "base")
RES = os.path.join(ROOT, "results", "T29")
SPLITS = os.path.join(RES, "splits.json")
ARMS = ["D", "SGC-P", "SGC-S", "SGC-B", "LR"]
FIRST = ("Amazon_clothing", 0)
OUT_FILES = ("run.json", "per_episode.npz", "pretrain.json")


def bdir(ds, k, s):
    return os.path.join(R22B, f"{ds}_5w{k}s", f"seed{s}")


# ---------------------------------------------------------------- §2.1 split restoration + gate Gs
def restore_splits():
    out, gs = {}, {"1": [], "2": [], "3": [], "4": [], "5": []}
    for ds in DATASETS:
        lab0 = np.load(os.path.join(bdir(ds, 1, 0), "labels.npy"))
        test0 = sorted(int(c) for c in json.load(open(os.path.join(bdir(ds, 1, 0), "split.json")))["class_list_test"])
        for k in SHOTS:
            for s in (0, 1, 2):
                if not np.array_equal(np.load(os.path.join(bdir(ds, k, s), "labels.npy")), lab0):
                    gs["5"].append(f"labels {ds} {k} {s}")
                t = sorted(int(c) for c in json.load(open(os.path.join(bdir(ds, k, s), "split.json")))["class_list_test"])
                if t != test0:
                    gs["5"].append(f"test {ds} {k} {s}")
        all_cls = sorted(int(c) for c in np.unique(lab0))
        out[ds] = {"test": test0, "n_nodes": int(len(lab0)), "n_classes": len(all_cls), "seeds": {}}
        for s in SEEDS:
            vk = {}
            for k in SHOTS:
                v = set()
                for f in ("eval_logits.npz", "fixed_eval_logits.npz"):
                    Z = np.load(os.path.join(bdir(ds, k, s), f))
                    v |= set(int(c) for c in np.unique(Z["classes"][Z["mode"] == 0]))
                vk[k] = sorted(v)
            valid = sorted(set(vk[1]) | set(vk[5]))
            base = sorted(set(all_cls) - set(valid) - set(test0))
            out[ds]["seeds"][str(s)] = {"base": base, "valid": valid, "valid_1shot": vk[1], "valid_5shot": vk[5]}
            if len(valid) != VALID_NUM[ds]:
                gs["2"].append(f"{ds} {s}: {len(valid)}")
            if len(set(vk[1]) | set(vk[5])) != VALID_NUM[ds]:
                gs["4"].append(f"{ds} {s}: 1-shot {len(vk[1])}, 5-shot {len(vk[5])}, union {len(valid)}")
            if set(base) & set(valid) or set(base) & set(test0) or set(valid) & set(test0):
                gs["3"].append(f"{ds} {s}")
            if s <= 2:
                for k in SHOTS:
                    sp = json.load(open(os.path.join(bdir(ds, k, s), "split.json")))
                    ref = {g: set(int(c) for c in sp[key]) for g, key in (("base", "class_list_train"), ("valid", "class_list_valid"), ("test", "class_list_test"))}
                    if ref != {"base": set(base), "valid": set(valid), "test": set(test0)}:
                        gs["1"].append(f"{ds} {k} {s}")
    os.makedirs(RES, exist_ok=True)
    rec = {"splits": out, "Gs": gs, "Gs_pass": not any(gs.values())}
    json.dump(rec, open(SPLITS, "w"), indent=1)
    print(json.dumps({"Gs": gs, "Gs_pass": rec["Gs_pass"],
                      "sizes": {ds: {s: [len(out[ds]["seeds"][s][g]) for g in ("base", "valid")] + [len(out[ds]["test"])] for s in out[ds]["seeds"]}
                                for ds in DATASETS}}))
    return rec["Gs_pass"]


# ---------------------------------------------------------------- features / linear head
def sgc_features(ds):
    """Â²(row-L1 X), no post-normalization. Matrix construction as fusion_baseline.diffusion_features_hops."""
    import scipy.sparse as sp
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    import fusion_baseline as fb
    from utils import load_data

    cwd = os.getcwd()
    os.chdir(fb.ROOT)
    try:
        edges, _, X, _, _, _, _, _, _, _ = load_data(ds)
    finally:
        os.chdir(cwd)
    X = X.numpy().astype(np.float32)
    rs = X.sum(1, keepdims=True)
    X = (X / np.where(rs == 0, 1, rs)).astype(np.float32)  # row-L1; zero-sum rows unchanged (pilot:113-114)
    ei, w = gcn_norm(edges, None, X.shape[0], add_self_loops=True)
    ei, w = ei.numpy(), w.numpy()
    A = sp.csr_matrix((w, (ei[1], ei[0])), shape=(X.shape[0], X.shape[0]))
    D = X
    for _ in range(2):
        D = (A @ D).astype(np.float32)
    return D


def fit_linear(W, b, Xs, ys, epochs=100, lr=0.2, wd=5e-6):
    import torch
    import torch.nn.functional as F

    W = W.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(Xs @ W.T + b, ys)
        loss.backward()
        opt.step()
    return W.detach(), b.detach(), float(loss.detach())


def init_linear(d, gen, dev):
    import torch

    bound = 1 / np.sqrt(d)
    W = (torch.rand(N_WAY, d, generator=gen) * 2 - 1) * bound
    b = (torch.rand(N_WAY, generator=gen) * 2 - 1) * bound
    return W.to(dev), b.to(dev)


def pretrain(Vt, base_cls, by_class, seed, dev):
    import torch
    import torch.nn.functional as F

    gen = torch.Generator().manual_seed(18700 + seed)
    prng = random.Random(18600 + seed)
    W, b = init_linear(Vt.shape[1], gen, dev)
    losses = []
    for _ in range(ITER):
        sel = prng.sample(base_cls, N_WAY)
        pools = [by_class[c] for c in sel]
        for _ in range(STEP):
            idx, lab = [], []
            for j, p in enumerate(pools):
                ids = prng.sample(p, TRAIN_SHOT)
                idx += ids
                lab += [j] * TRAIN_SHOT
            ys = torch.tensor(lab, device=dev)
            W, b, l = fit_linear(W, b, Vt[idx], ys)
            losses.append(l)
    with torch.no_grad():
        z = Vt[idx] @ W.T + b
        last = {"loss": float(F.cross_entropy(z, ys)), "acc": float((z.argmax(1) == ys).float().mean())}
    return W, b, losses, last


def load_eps(ds, k, s):
    Z = np.load(os.path.join(bdir(ds, k, s), "fixed_eval_logits.npz"))
    m = np.where(Z["mode"] == 1)[0]
    m = m[np.argsort(Z["ep_idx"][m], kind="stable")]
    return Z["support"][m], Z["query"][m], Z["query_y"][m], Z["ep_idx"][m]


def one(ds, seed):
    import torch
    from sklearn.linear_model import LogisticRegression

    import fusion_baseline as fb

    t0 = time.time()
    torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sp = json.load(open(SPLITS))
    assert sp["Gs_pass"]
    labels = np.load(os.path.join(bdir(ds, 1, 0), "labels.npy"))
    base_cls = sp["splits"][ds]["seeds"][str(seed)]["base"]
    by_class = {c: np.where(labels == c)[0].tolist() for c in base_cls}
    V_d = fb.diffusion_features(ds)
    V_s = sgc_features(ds)
    Vt = torch.from_numpy(V_s).to(dev)
    d = V_s.shape[1]
    t_feat = time.time() - t0
    print(f"{ds} seed{seed} features {t_feat:.1f}s dim {d} device {dev}", flush=True)
    t1 = time.time()
    W_pre, b_pre, losses, last = pretrain(Vt, base_cls, by_class, seed, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_pre = time.time() - t1
    print(f"{ds} seed{seed} pretrain {t_pre:.1f}s last loss {last['loss']:.4f} acc {last['acc']:.4f}", flush=True)
    out_dir = os.path.join(RES, ds, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"dataset": ds, "seed": seed, "n_base": len(base_cls), "base": base_cls, "round_last_epoch_loss": losses,
               "last_round": last, "all_finite": bool(np.all(np.isfinite(losses))), "t_pretrain_s": t_pre},
              open(os.path.join(out_dir, "pretrain.json"), "w"))
    npz, summ = {}, {}
    for k in SHOTS:
        t2 = time.time()
        sup, qry, qy, ep_idx = load_eps(ds, k, seed)
        yq = np.repeat(np.arange(N_WAY), qry.shape[1] // N_WAY)
        ys = torch.tensor(np.repeat(np.arange(N_WAY), k), device=dev)
        gen = torch.Generator().manual_seed((18800 if k == 1 else 18900) + seed)
        pred = {a: [] for a in ARMS}
        for a, V in (("D", V_d), ("SGC-P", V_s)):
            P = V[sup].reshape(len(sup), N_WAY, k, -1).mean(2)
            P = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
            pred[a] = np.einsum("eqd,end->eqn", V[qry], P).argmax(-1)
            assert np.array_equal((pred[a] == yq[None]).mean(-1), fb.cosine_probe_acc(V, sup, qry, k))
        for e in range(len(sup)):
            s_t, q_t = torch.as_tensor(sup[e], device=dev), torch.as_tensor(qry[e], device=dev)
            Wr, br, _ = fit_linear(*init_linear(d, gen, dev), Vt[s_t], ys)
            pred["SGC-S"].append((Vt[q_t] @ Wr.T + br).argmax(1).cpu().numpy())
            Wf, bf, _ = fit_linear(W_pre, b_pre, Vt[s_t], ys)
            pred["SGC-B"].append((Vt[q_t] @ Wf.T + bf).argmax(1).cpu().numpy())
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(V_d[sup[e]], np.repeat(np.arange(N_WAY), k))
            pred["LR"].append(clf.predict(V_d[qry[e]]))
        summ[k] = {"qy_class_order": bool((qy == yq[None]).all()), "n_eps": int(len(sup)), "sec": time.time() - t2}
        for a in ARMS:
            p = np.asarray(pred[a])
            npz[f"pred_{k}_{a}"] = p
            npz[f"acc_{k}_{a}"] = (p == yq[None]).mean(-1)
            summ[k][a] = 100 * float(npz[f"acc_{k}_{a}"].mean())
        npz[f"ep_idx_{k}"] = ep_idx
        print(f"{ds} seed{seed} {k}-shot " + " ".join(f"{a} {summ[k][a]:.2f}" for a in ARMS) + f" ({summ[k]['sec']:.1f}s)", flush=True)
    np.savez(os.path.join(out_dir, "per_episode.npz"), **npz)
    json.dump({"dataset": ds, "seed": seed, "device": str(dev), "dim": d, "t_feat_s": t_feat, "t_pretrain_s": t_pre,
               "t_total_inproc_s": time.time() - t0, "shots": summ}, open(os.path.join(out_dir, "run.json"), "w"), indent=1)


# ---------------------------------------------------------------- driver (as tools/run_t27.py)
def poll_mem(gpu, stop, box):
    while not stop.is_set():
        try:
            v = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
            box["max_mib"] = max(box["max_mib"], int(v.split()[0]))
            box["n"] += 1
        except Exception:
            box["err"] += 1
        stop.wait(1.0)


def run_once(gpu, ds, s, log_name):
    from run_t21 import GPUS

    out = os.path.join(RES, ds, f"seed{s}")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu], PYTHONUNBUFFERED="1")
    stop, box = threading.Event(), {"max_mib": 0, "n": 0, "err": 0}
    th = threading.Thread(target=poll_mem, args=(gpu, stop, box), daemon=True)
    th.start()
    t0 = time.time()
    with open(os.path.join(out, log_name), "w") as log:
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "one", ds, str(s)], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    t1 = time.time()
    stop.set()
    th.join()
    tag = log_name[len("console"):-len(".log")]
    json.dump({"gpu": gpu, "uuid": GPUS[gpu], **box}, open(os.path.join(out, f"gpu_mem{tag}.json"), "w"))
    ok = rc == 0 and all(os.path.exists(os.path.join(out, f)) for f in OUT_FILES)
    return ok, rc, t0, t1, box["max_mib"]


def worker(gpu, q, lock):
    while True:
        try:
            ds, s = q.get_nowait()
        except queue.Empty:
            return
        attempts = []
        for log_name in ("console.log", "console_retry1.log"):
            ok, rc, t0, t1, mem = run_once(gpu, ds, s, log_name)
            attempts.append({"log": log_name, "exit": rc, "ok": ok, "start": t0, "end": t1, "sec": t1 - t0, "gpu_max_mib": mem})
            if ok:
                break
        rec = {"dataset": ds, "seed": s, "gpu": gpu, "ok": attempts[-1]["ok"], "attempts": attempts}
        with lock:
            with open(os.path.join(RES, "_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {ds} seed{s} ok={rec['ok']} tries={len(attempts)} "
                  f"sec={attempts[-1]['sec']:.1f} gpu_max_mib={attempts[-1]['gpu_max_mib']}", flush=True)


def main(jobs, gpus):
    os.makedirs(RES, exist_ok=True)
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
    if sys.argv[1] == "splits":
        restore_splits()
    elif sys.argv[1] == "one":
        one(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "phase1":
        main([FIRST], [0])
    elif sys.argv[1] == "phase2":
        main([(ds, s) for ds in DATASETS for s in SEEDS if (ds, s) != FIRST], [0, 1, 2, 4, 5, 6])
