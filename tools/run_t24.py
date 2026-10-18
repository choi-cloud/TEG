"""T24: train the official GPN (kaize0409/GPN_Graph-Few-shot @ 40ef635, tools/gpn/) on the T22 `base` split.

Training step = upstream train_gpn.py train() formula (valuator-weighted prototype, euclidean, log_softmax, NLL; two Adam,
lr 0.005, wd 5e-4), 1000 episodes, 5-way k-shot, query 20 / class. Episodes: upstream task_generator rule with dedicated
generators (train Random(18000 + s), valid pool 50 Random(18100 + s), test pool 50 Random(18200 + s)); model init
torch.manual_seed(18300 + s). Selection (not in upstream): after each completed episode e in {10, 20, ..., 1000}, mean
accuracy on the valid pool in eval mode; best = max (ties -> later); emb_best.npy / score_best.npy / ckpt_best.pt saved then.
Data: features / adj / degree from utils.load_data (its split is discarded); classes and labels from
results/T22/base/<ds>_5w<k>s/seed<s>/{split.json, labels.npy}.
Usage  python tools/run_t24.py hc                              -> Hc synthetic check (needs /tmp/gpn_upstream)
       python tools/run_t24.py one --ds <ds> --k <k> --seed <s> -> one run (device cuda:0)
       python tools/run_t24.py                                 -> 18 runs on GPUs 0, 1, 2, 4, 5, 6 (UUID)
"""
import argparse
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "gpn"))  # upstream models.py does `from layers import ...`; must win over ROOT/layers
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from models import GPN_Encoder, GPN_Valuator  # noqa: E402

assert os.path.abspath(sys.modules["layers"].__file__) == os.path.join(ROOT, "tools", "gpn", "layers.py")
sys.path.insert(1, ROOT)
sys.path.insert(2, os.path.join(ROOT, "tools"))
from run_t21 import DATASETS, GPUS, SEEDS, SHOTS  # noqa: E402
from utils import euclidean_dist  # noqa: E402  (same as upstream utils.euclidean_dist)

N_WAY, N_QRY, N_EPISODES, EVAL_EVERY, N_POOL = 5, 20, 1000, 10, 50
LR, WD, HIDDEN, DROPOUT = 0.005, 5e-4, 16, 0.5
RESULTS = os.path.join(ROOT, "results", "T24", "gpn")
T22 = os.path.join(ROOT, "results", "T22", "base")
UPSTREAM = "/tmp/gpn_upstream"


def load_inputs(ds, k, s):
    from utils import load_data

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        _, adj, features, labels_ld, degree, _, _, _, _, _ = load_data(ds)
    finally:
        os.chdir(cwd)
    sp_path = os.path.join(T22, f"{ds}_5w{k}s", f"seed{s}", "split.json")
    sp = json.load(open(sp_path))
    labels = np.load(os.path.join(os.path.dirname(sp_path), "labels.npy"))
    by_class = {int(c): np.where(labels == int(c))[0].tolist() for g in sp for c in sp[g]}
    meta = {"split_json": os.path.relpath(sp_path, ROOT), "split_sha256": hashlib.sha256(open(sp_path, "rb").read()).hexdigest(),
            "labels_equal_load_data": bool(np.array_equal(labels_ld.numpy(), labels))}
    return adj, features, degree, torch.from_numpy(labels).long(), sp, by_class, meta


def task(rng, id_by_class, class_list, n_way, k_shot, m_query):
    """upstream utils.task_generator with random -> rng."""
    class_selected = rng.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = rng.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])
    return np.array(id_support), np.array(id_query), class_selected


def gpn_logp(embeddings, scores, degrees, id_support, id_query, n_way, k_shot):
    """upstream train()/test(): node importance -> weighted prototype -> euclidean -> log_softmax."""
    z_dim = embeddings.size()[1]
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    return F.log_softmax(-dists, dim=1)


def query_targets(labels_cpu, class_selected, id_query):
    return torch.LongTensor([class_selected.index(int(i)) for i in labels_cpu[id_query]])


def accuracy(output, labels):  # upstream utils.accuracy
    preds = output.max(1)[1].type_as(labels)
    return float(preds.eq(labels).double().sum() / len(labels))


def train_step(encoder, scorer, opt_e, opt_s, features, adj, degrees, labels_cpu, class_selected, id_support, id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    opt_e.zero_grad()
    opt_s.zero_grad()
    embeddings = encoder(features, adj)
    scores = scorer(features, adj)
    output = gpn_logp(embeddings, scores, degrees, id_support, id_query, n_way, k_shot)
    labels_new = query_targets(labels_cpu, class_selected, id_query).to(output.device)
    loss = F.nll_loss(output, labels_new)
    loss.backward()
    opt_e.step()
    opt_s.step()
    return float(loss), accuracy(output.detach().cpu(), labels_new.cpu())


@torch.no_grad()
def evaluate(encoder, scorer, features, adj, degrees, labels_cpu, pool, n_way, k_shot):
    """upstream test() on every task of the pool; embeddings / scores computed once (eval mode is deterministic)."""
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    scores = scorer(features, adj)
    accs, f1s = [], []
    for id_support, id_query, class_selected in pool:
        output = gpn_logp(embeddings, scores, degrees, id_support, id_query, n_way, k_shot).cpu()
        y = query_targets(labels_cpu, class_selected, id_query)
        accs.append(accuracy(output, y))
        f1s.append(float(f1_score(y, output.max(1)[1].type_as(y), average="weighted")))
    return float(np.mean(accs)), float(np.mean(f1s)), embeddings, scores


def build_models(nfeat, s):
    torch.manual_seed(18300 + s)
    encoder = GPN_Encoder(nfeat=nfeat, nhid=HIDDEN, dropout=DROPOUT)
    scorer = GPN_Valuator(nfeat=nfeat, nhid=HIDDEN, dropout=DROPOUT)
    return encoder, scorer


def train_one(ds, k, s):
    t0 = time.time()
    dev = torch.device("cuda:0")
    out = os.path.join(RESULTS, f"{ds}_5w{k}s", f"seed{s}")
    os.makedirs(out, exist_ok=True)
    adj, features, degree, labels_cpu, sp, by_class, meta = load_inputs(ds, k, s)
    encoder, scorer = build_models(features.shape[1], s)
    opt_e = optim.Adam(encoder.parameters(), lr=LR, weight_decay=WD)
    opt_s = optim.Adam(scorer.parameters(), lr=LR, weight_decay=WD)
    encoder.to(dev)
    scorer.to(dev)
    features, adj, degrees = features.to(dev), adj.to(dev), degree.to(dev)
    tr, va, te = sp["class_list_train"], sp["class_list_valid"], sp["class_list_test"]
    rv, rt, rtr = random.Random(18100 + s), random.Random(18200 + s), random.Random(18000 + s)
    valid_pool = [task(rv, by_class, va, N_WAY, k, N_QRY) for _ in range(N_POOL)]
    test_pool = [task(rt, by_class, te, N_WAY, k, N_QRY) for _ in range(N_POOL)]
    curves = {"episode": [], "valid_acc": [], "valid_f1": [], "test_acc": [], "test_f1": [], "train_acc_ma100": [], "train_acc": [],
              "train_loss": []}
    best = {"episode": None, "valid_acc": -1.0}
    for ep in range(1, N_EPISODES + 1):
        id_support, id_query, class_selected = task(rtr, by_class, tr, N_WAY, k, N_QRY)
        loss, acc = train_step(encoder, scorer, opt_e, opt_s, features, adj, degrees, labels_cpu, class_selected, id_support, id_query,
                               N_WAY, k)
        curves["train_acc"].append(acc)
        curves["train_loss"].append(loss)
        if ep % EVAL_EVERY == 0:
            v_acc, v_f1, emb, sc = evaluate(encoder, scorer, features, adj, degrees, labels_cpu, valid_pool, N_WAY, k)
            t_acc, t_f1, _, _ = evaluate(encoder, scorer, features, adj, degrees, labels_cpu, test_pool, N_WAY, k)
            for key, v in (("episode", ep), ("valid_acc", v_acc), ("valid_f1", v_f1), ("test_acc", t_acc), ("test_f1", t_f1),
                           ("train_acc_ma100", float(np.mean(curves["train_acc"][-100:])))):
                curves[key].append(v)
            print(f"ep {ep} loss {loss:.4f} train_ma100 {curves['train_acc_ma100'][-1]:.4f} valid {v_acc:.4f} test {t_acc:.4f}", flush=True)
            if v_acc >= best["valid_acc"]:  # ties -> later episode
                best = {"episode": ep, "valid_acc": v_acc, "test_acc": t_acc, "test_f1": t_f1}
                np.save(os.path.join(out, "emb_best.npy"), emb.float().cpu().numpy())
                np.save(os.path.join(out, "score_best.npy"), sc.view(-1).float().cpu().numpy())
                torch.save({"encoder": encoder.state_dict(), "scorer": scorer.state_dict(), "episode": ep}, os.path.join(out, "ckpt_best.pt"))
    json.dump(curves, open(os.path.join(out, "curves.json"), "w"))
    rec = {"dataset": ds, "shot": k, "seed": s, "best_episode": best["episode"], "best_valid_acc": best["valid_acc"],
           "test_acc_at_best": best["test_acc"], "test_f1_at_best": best["test_f1"], "train_sec": time.time() - t0,
           "config": {"way": N_WAY, "qry": N_QRY, "episodes": N_EPISODES, "eval_every": EVAL_EVERY, "pool": N_POOL, "lr": LR, "wd": WD,
                      "hidden": HIDDEN, "dropout": DROPOUT, "seed_offsets": {"train": 18000, "valid": 18100, "test": 18200, "init": 18300}},
           **meta}
    json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1)
    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------- Hc (synthetic, upstream train()/test() executed verbatim)
def hc_check():
    import ast
    import copy
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location("gpn_up_utils", os.path.join(UPSTREAM, "utils.py"))
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    src = open(os.path.join(UPSTREAM, "train_gpn.py")).read()
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("train", "test")]
    captured = []

    def nll_shim(output, target, *a, **kw):
        captured.append((output.detach().clone(), F.nll_loss(output, target, *a, **kw)))
        return captured[-1][1]

    F_shim = types.SimpleNamespace(**{n: getattr(F, n) for n in dir(F) if not n.startswith("__")})
    F_shim.nll_loss = nll_shim

    g = torch.Generator().manual_seed(0)
    n, d, n_cls = 50, 8, 4
    A = (torch.rand(n, n, generator=g) < 0.1).float()
    A = ((A + A.T) > 0).float()
    A.fill_diagonal_(0)
    import scipy.sparse as sp_

    adj = up.sparse_mx_to_torch_sparse_tensor(up.normalize_adj(sp_.coo_matrix(A.numpy()) + sp_.eye(n)))
    features = torch.rand(n, d, generator=g)
    labels = torch.arange(n) % n_cls
    degrees = torch.randint(1, 6, (n, 1), generator=g).float()
    by_class = {c: np.where(labels.numpy() == c)[0].tolist() for c in range(n_cls)}
    torch.manual_seed(7)
    enc0, sc0 = GPN_Encoder(d, HIDDEN, DROPOUT), GPN_Valuator(d, HIDDEN, DROPOUT)
    enc_u, sc_u, enc_m, sc_m = copy.deepcopy(enc0), copy.deepcopy(sc0), copy.deepcopy(enc0), copy.deepcopy(sc0)
    ou_e, ou_s = optim.Adam(enc_u.parameters(), lr=LR, weight_decay=WD), optim.Adam(sc_u.parameters(), lr=LR, weight_decay=WD)
    om_e, om_s = optim.Adam(enc_m.parameters(), lr=LR, weight_decay=WD), optim.Adam(sc_m.parameters(), lr=LR, weight_decay=WD)
    ns = {"torch": torch, "np": np, "F": F_shim, "args": types.SimpleNamespace(cuda=False), "encoder": enc_u, "scorer": sc_u,
          "optimizer_encoder": ou_e, "optimizer_scorer": ou_s, "features": features, "adj": adj, "labels": labels, "degrees": degrees,
          "euclidean_dist": up.euclidean_dist, "accuracy": up.accuracy, "f1": up.f1}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "train_gpn.py", "exec"), ns)
    rng = random.Random(1)
    way, shot, q = 2, 1, 2
    res = {"loss_absdiff": [], "param_absdiff": [], "test_logp_absdiff": [], "test_acc_absdiff": []}
    for step in range(3):
        sup, qry, cls = task(rng, by_class, list(range(n_cls)), way, shot, q)
        torch.manual_seed(100 + step)
        ns["train"](cls, sup, qry, way, shot)
        loss_u = float(captured[-1][1])
        torch.manual_seed(100 + step)
        loss_m, _ = train_step(enc_m, sc_m, om_e, om_s, features, adj, degrees, labels, cls, sup, qry, way, shot)
        res["loss_absdiff"].append(abs(loss_u - loss_m))
        pd = [float((a - b).abs().max()) for a, b in zip(list(enc_u.parameters()) + list(sc_u.parameters()),
                                                          list(enc_m.parameters()) + list(sc_m.parameters()))]
        res["param_absdiff"].append(max(pd))
        sup2, qry2, cls2 = task(rng, by_class, list(range(n_cls)), way, shot, q)
        acc_u, _ = ns["test"](cls2, sup2, qry2, way, shot)
        logp_u = captured[-1][0]
        acc_m, _, emb, sc = evaluate(enc_m, sc_m, features, adj, degrees, labels, [(sup2, qry2, cls2)], way, shot)
        logp_m = gpn_logp(emb, sc, degrees, sup2, qry2, way, shot)
        res["test_logp_absdiff"].append(float((logp_u - logp_m).abs().max()))
        res["test_acc_absdiff"].append(abs(float(acc_u) - acc_m))
    mx = {k: max(v) for k, v in res.items()}
    res["max"] = mx
    res["pass"] = all(v <= 1e-6 for v in mx.values())
    os.makedirs(os.path.join(ROOT, "results", "T24"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "T24", "hc_check.json"), "w"), indent=1)
    print(json.dumps(res))


# ---------------------------------------------------------------- driver (tools/run_t21.py pattern)
def jobs():
    for s in SEEDS:
        for ds in DATASETS:
            for k in SHOTS:
                yield ds, k, s


def run_once(gpu, ds, k, s, log_name):
    out = os.path.join(RESULTS, f"{ds}_5w{k}s", f"seed{s}")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join("tools", "run_t24.py"), "one", "--ds", ds, "--k", str(k), "--seed", str(s)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu])
    t0 = time.time()
    with open(os.path.join(out, log_name), "w") as log:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = rc == 0 and all(os.path.exists(os.path.join(out, f)) for f in ("run.json", "emb_best.npy", "score_best.npy", "ckpt_best.pt", "curves.json"))
    return ok, rc, t0, time.time()


def worker(gpu, q, lock):
    while True:
        try:
            ds, k, s = q.get_nowait()
        except queue.Empty:
            return
        attempts = []
        for log_name in ("console.log", "console_retry1.log"):
            ok, rc, t0, t1 = run_once(gpu, ds, k, s, log_name)
            attempts.append({"log": log_name, "exit": rc, "ok": ok, "start": t0, "end": t1, "sec": t1 - t0, "over_10min": t1 - t0 > 600})
            if ok:
                break
        rec = {"dataset": ds, "shot": k, "seed": s, "gpu": gpu, "ok": attempts[-1]["ok"], "attempts": attempts}
        with lock:
            with open(os.path.join(RESULTS, "_runs.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            warn = " OVER_10MIN" if attempts[-1]["over_10min"] else ""
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {ds} 5w{k}s seed{s} ok={rec['ok']} tries={len(attempts)} "
                  f"sec={attempts[-1]['sec']:.1f}{warn}", flush=True)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    q = queue.Queue()
    for j in jobs():
        q.put(j)
    lock = threading.Lock()
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(gpu, q, lock)) for gpu in GPUS]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", choices=["all", "one", "hc"])
    ap.add_argument("--ds")
    ap.add_argument("--k", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.mode == "hc":
        hc_check()
    elif a.mode == "one":
        train_one(a.ds, a.k, a.seed)
    else:
        main()
