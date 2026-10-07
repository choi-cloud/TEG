"""T16a erasure measurement functions (no side effects on import: no chdir, no sys.path change, no RNG).

- knn_purity: same computation as tools/analyze_t14.knn_purity (row L2-normalized cosine, top-k inside the group, float64);
  `device` only moves the computation (None = CPU).
- episodes / fewshot_acc: same computation as tools/analyze_t06b.episodes / fewshot_acc (5-way, query 5, cosine prototypes),
  with the number of episodes as an argument.
- proto_euc_acc / proto_cos_acc / lr_acc: per-episode accuracies on fixed episodes (support/query in class order).
- lr_acc: sklearn LogisticRegression(C=1.0, lbfgs, max_iter=100) on row L2-normalized features (deterministic).
"""
import warnings

import numpy as np
import torch

N_WAY, N_QRY, BATCH = 5, 5, 50


def l2n(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def knn_purity(Z, nodes, y, k=10, block=4096, device=None):
    """Z: [N, d] numpy or torch; nodes: group node ids; y: their labels (numpy)."""
    Zt = Z if torch.is_tensor(Z) else torch.from_numpy(np.asarray(Z))
    V = torch.nn.functional.normalize(Zt[torch.as_tensor(nodes, device=Zt.device)].double(), dim=1)
    if device is not None:
        V = V.to(device)
    yy = torch.as_tensor(y, device=V.device)
    out = []
    for s in range(0, len(nodes), block):
        sim = V[s : s + block] @ V.T
        r = torch.arange(sim.shape[0], device=V.device)
        sim[r, r + s] = float("-inf")
        nb = sim.topk(k, dim=1).indices
        out.append((yy[nb] == yy[s : s + block, None]).double().mean(1))
    return float(torch.cat(out).mean())


def episodes(rng, classes, by_class, k, n_epi):
    out = []
    for _ in range(n_epi):
        cls = rng.sample(classes, N_WAY)
        sup, qry = [], []
        for c in cls:
            ids = rng.sample(by_class[c], k + N_QRY)
            sup.append(ids[:k])
            qry.append(ids[k:])
        out.append((np.array(sup).reshape(-1), np.array(qry).reshape(-1)))
    return out


def _probe_cos(V, sup, qry, k):
    B = sup.shape[0]
    P = V[sup].reshape(B, N_WAY, k, V.shape[1]).mean(2)
    Q = V[qry].reshape(B, -1, V.shape[1])
    score = np.einsum("bqd,bnd->bqn", l2n(Q), l2n(P))
    true = np.repeat(np.arange(N_WAY), qry.shape[1] // N_WAY)[None, :]
    return (score.argmax(2) == true).mean(1)


def fewshot_acc(V, eps, k):
    accs = []
    for b in range(0, len(eps), BATCH):
        sup = np.stack([e[0] for e in eps[b : b + BATCH]])
        qry = np.stack([e[1] for e in eps[b : b + BATCH]])
        accs.append(_probe_cos(V, sup, qry, k))
    return 100 * float(np.concatenate(accs).mean())


def _stack(eps):
    return np.stack([np.asarray(e[0]) for e in eps]), np.stack([np.asarray(e[1]) for e in eps])


def proto_euc_acc(V, eps, k):
    """Per-episode accuracy of euclidean prototypes (support mean), argmin squared distance (same rule as L_G)."""
    sup, qry = _stack(eps)
    B = sup.shape[0]
    V = V.astype(np.float64)
    P = V[sup].reshape(B, N_WAY, k, V.shape[1]).mean(2)
    Q = V[qry].reshape(B, -1, V.shape[1])
    d = ((Q[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1)
    true = np.repeat(np.arange(N_WAY), qry.shape[1] // N_WAY)[None, :]
    return (d.argmin(2) == true).mean(1)


def proto_cos_acc(V, eps, k):
    sup, qry = _stack(eps)
    return _probe_cos(l2n(V.astype(np.float64)), sup, qry, k)


def lr_acc(V, eps, k):
    from sklearn.linear_model import LogisticRegression

    V = l2n(V.astype(np.float64))
    ys = np.repeat(np.arange(N_WAY), k)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for e in eps:
            s, q = e[0], e[1]
            s, q = np.asarray(s), np.asarray(q)
            yq = np.repeat(np.arange(N_WAY), len(q) // N_WAY)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=100).fit(V[s], ys)
            out.append(float((clf.predict(V[q]) == yq).mean()))
    return np.array(out)
