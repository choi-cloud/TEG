"""Sandbox pilot (NOT a registered T## experiment): where does the gap between the published SGC baseline
(Meta-GNN protocol) and our training-free D come from?

Arms, all on the same fixed test episodes (5-way, k-shot, 20 queries/class, test classes from *_test.mat):
  D            : row-L2(Â^2 X_raw), cosine to support-mean prototype. Zero parameters. (our D)
  D_l1         : same prototype rule on Meta-GNN features Â^2 rowL1(X) (isolates feature preprocessing)
  LR_scratch   : sklearn LogisticRegression(C=1) fit on support only, on D features (T21 J4 head)
  SGC_scratch  : Meta-GNN SGC head (nn.Linear + bias, Adam lr 0.2, wd 5e-6, 100 epochs) fit on support only,
                 random init, on Meta-GNN features
  SGC_metagnn  : same head, but initialized from a base-class "pretraining" exactly as sgc_fewshot/citation.py
                 (iteration x step rounds; each round: random N base classes -> labels 0..N-1, train_shot=20 per class,
                 100 epochs on the SAME linear layer), then fine-tuned on the support. N-way generalization of the
                 public 2-way code.
Usage: python -I pilot_sgc_vs_d.py <data_dir> <dataset> <seed> [n_eps]
"""
import json
import random
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

VALID_NUM = {"Amazon_clothing": 17, "Amazon_eletronics": 36, "dblp": 27}
N_WAY, N_QRY = 5, 20
SHOTS = [1, 3, 5]


def load(data_dir, ds):
    n1, n2 = [], []
    for line in open(f"{data_dir}/{ds}_network"):
        a, b = line.strip().split("\t")
        n1.append(int(a)); n2.append(int(b))
    N = max(max(n1), max(n2)) + 1
    A = sp.coo_matrix((np.ones(len(n1)), (n1, n2)), shape=(N, N))
    tr = sio.loadmat(f"{data_dir}/{ds}_train.mat"); te = sio.loadmat(f"{data_dir}/{ds}_test.mat")
    d = tr["Attributes"].shape[1]
    X = sp.lil_matrix((N, d), dtype=np.float32)
    X[tr["Index"].ravel()] = tr["Attributes"]; X[te["Index"].ravel()] = te["Attributes"]
    X = X.tocsr()
    y = np.zeros(N, dtype=np.int64)
    y[tr["Index"].ravel()] = tr["Label"].ravel(); y[te["Index"].ravel()] = te["Label"].ravel()
    train_cls = sorted(set(tr["Label"].ravel().tolist())); test_cls = sorted(set(te["Label"].ravel().tolist()))
    # Â = D^-1/2 (A+I) D^-1/2 as in GPN/TEG utils.normalize_adj (row sums of A+I). Meta-GNN AugNormAdj is the same form.
    A = A + sp.eye(N)
    r = np.asarray(A.sum(1)).ravel(); dinv = np.power(r, -0.5); dinv[np.isinf(dinv)] = 0
    Ah = sp.diags(dinv) @ A @ sp.diags(dinv)
    Ah = Ah.tocsr().astype(np.float32)
    return Ah, X, y, train_cls, test_cls


def prop2(Ah, X):
    return np.asarray((Ah @ (Ah @ X)).todense() if sp.issparse(X) else Ah @ (Ah @ X), dtype=np.float32)


def l2n(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def episodes(rng, classes, by_class, k, n):
    out = []
    for _ in range(n):
        cls = rng.sample(classes, N_WAY)
        s, q = [], []
        for c in cls:
            ids = rng.sample(by_class[c], k + N_QRY)
            s += ids[:k]; q += ids[k:]
        out.append((np.array(s), np.array(q)))
    return out


def proto_acc(V, s, q, k):
    P = l2n(V[s].reshape(N_WAY, k, -1).mean(1)); Q = l2n(V[q])
    yq = np.repeat(np.arange(N_WAY), N_QRY)
    return float(((Q @ P.T).argmax(1) == yq).mean())


def fit_linear(W, b, Xs, ys, epochs=100, lr=0.2, wd=5e-6):
    W = W.clone().requires_grad_(True); b = b.clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(Xs @ W.T + b, ys).backward()
        opt.step()
    return W.detach(), b.detach()


def init_linear(d, gen):
    # nn.Linear default init: U(-1/sqrt(d), 1/sqrt(d))
    bound = 1 / np.sqrt(d)
    W = (torch.rand(N_WAY, d, generator=gen) * 2 - 1) * bound
    b = (torch.rand(N_WAY, generator=gen) * 2 - 1) * bound
    return W, b


def main():
    data_dir, ds, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_eps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    t0 = time.time()
    Ah, X, y, train_cls, test_cls = load(data_dir, ds)
    random.seed(seed)
    valid_cls = random.sample(train_cls, VALID_NUM[ds])
    base_cls = sorted(set(train_cls) - set(valid_cls))
    by_class = {}
    for i, c in enumerate(y.tolist()):
        by_class.setdefault(c, []).append(i)

    V_d = l2n(prop2(Ah, X))                          # our D features
    rs = np.asarray(X.sum(1)).ravel(); rs[rs == 0] = 1
    X_l1 = sp.diags(1.0 / rs) @ X                    # Meta-GNN row_normalize (L1)
    V_m = prop2(Ah, X_l1.tocsr().astype(np.float32))  # Meta-GNN SGC features (no post-normalization)
    Vm_t = torch.from_numpy(V_m)
    d = V_m.shape[1]
    t_feat = time.time() - t0

    # Meta-GNN-style base pretraining of ONE linear layer (N-way generalization of sgc_fewshot/citation.py)
    gen = torch.Generator().manual_seed(18300 + seed)
    prng = random.Random(18000 + seed)
    W0, b0 = init_linear(d, gen)
    W, b = W0.clone(), b0.clone()
    ITER, STEP, TRAIN_SHOT = 50, 50, 20
    for _ in range(ITER):
        sel = prng.sample(base_cls, N_WAY)
        pools = [by_class[c] for c in sel]
        for _ in range(STEP):
            idx, lab = [], []
            for j, p in enumerate(pools):
                ids = prng.sample(p, TRAIN_SHOT); idx += ids; lab += [j] * TRAIN_SHOT
            W, b = fit_linear(W, b, Vm_t[idx], torch.tensor(lab))
    W_pre, b_pre = W, b
    # base-pretrain fit check: accuracy on one fresh base episode set
    t_pre = time.time() - t0 - t_feat

    res = {"dataset": ds, "seed": seed, "n_eps": n_eps, "dim": d, "n_base": len(base_cls), "n_test": len(test_cls),
           "t_feat_s": round(t_feat, 1), "t_pretrain_s": round(t_pre, 1), "shots": {}}
    for k in SHOTS:
        eps = episodes(random.Random(1000 + seed), test_cls, by_class, k, n_eps)
        acc = {a: [] for a in ["D", "D_l1", "LR_scratch", "SGC_scratch", "SGC_metagnn"]}
        yq = np.repeat(np.arange(N_WAY), N_QRY); ys = torch.tensor(np.repeat(np.arange(N_WAY), k))
        for e, (s, q) in enumerate(eps):
            acc["D"].append(proto_acc(V_d, s, q, k))
            acc["D_l1"].append(proto_acc(V_m, s, q, k))
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(V_d[s], np.repeat(np.arange(N_WAY), k))
            acc["LR_scratch"].append(float((lr.predict(V_d[q]) == yq).mean()))
            g = torch.Generator().manual_seed(18400 + 1000 * seed + e)
            Wr, br = fit_linear(*init_linear(d, g), Vm_t[s], ys)
            acc["SGC_scratch"].append(float(((Vm_t[q] @ Wr.T + br).argmax(1).numpy() == yq).mean()))
            Wf, bf = fit_linear(W_pre, b_pre, Vm_t[s], ys)
            acc["SGC_metagnn"].append(float(((Vm_t[q] @ Wf.T + bf).argmax(1).numpy() == yq).mean()))
        res["shots"][k] = {a: {"mean": 100 * float(np.mean(v)), "per_ep": [round(x, 4) for x in v]} for a, v in acc.items()}
        print(ds, seed, k, {a: round(100 * np.mean(v), 2) for a, v in acc.items()}, flush=True)
    res["t_total_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"/home/claude/pilot_sgc/out_{ds}_s{seed}.json", "w"))


if __name__ == "__main__":
    main()
