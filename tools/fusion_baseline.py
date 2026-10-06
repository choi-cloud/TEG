"""Dual-view fusion baseline (T08b): TEG view L x diffusion-cosine view D, calibrated and combined.

Standalone on purpose (to be copied to other branches): depends only on the repo's data loader (`utils.load_data`)
and `torch_geometric`'s `gcn_norm`. Does not import any tools/analyze_*.py. No global RNG calls.

Definitions (T08 §3.1, T08b §1.1):
- h_bar = row-normalized A_hat^2 X, A_hat = gcn_norm(edge_index, add_self_loops=True) (same as GCNConv).
- view D score: l^D_{q,c} = cos(h_bar_q, mean of support h_bar of class c).
- view L score: from the dumped accuracy() tensor p = softmax(-dist): "log" -> log(p), "prob" -> p (T08 reproduction).
- temperatures on valid (mean NLL): log10 T_L in {-2.0, -1.8, ..., 4.0}, log10 s in {0.0, 0.1, ..., 3.0}.
- combination: log p~ = a * log softmax(l^L / T_L) + (1 - a) * log softmax(s * l^D); p_hat = softmax(log p~).
- alpha: "val" = argmin valid NLL over A21 (ties -> closer to 1), or a fixed float.
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_WAY = 5
TL_GRID = 10.0 ** np.round(np.arange(-2.0, 4.0 + 1e-9, 0.2), 1)  # 31 values
S_GRID = 10.0 ** np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 1)  # 31 values
A21 = np.round(np.arange(0, 21) * 0.05, 2)
TINY = np.finfo(np.float32).tiny


def diffusion_features(dataset):
    """h_bar = row-normalized A_hat^2 X for `dataset` (load_data reads ./dataset relative to the repo root)."""
    from utils import load_data

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        edges, _, X, _, _, _, _, _, _, _ = load_data(dataset)
    finally:
        os.chdir(cwd)
    X = X.numpy().astype(np.float32)
    ei, w = gcn_norm(edges, None, X.shape[0], add_self_loops=True)
    ei, w = ei.numpy(), w.numpy()
    A = sp.csr_matrix((w, (ei[1], ei[0])), shape=(X.shape[0], X.shape[0]))  # out[target] += w * x[source]
    D2 = (A @ (A @ X)).astype(np.float32)
    return D2 / np.maximum(np.linalg.norm(D2, axis=1, keepdims=True), 1e-12)


def view_d_scores(H, support, query, k, n_way=N_WAY):
    """support (E, n_way*k), query (E, n_way*Q) global ids in class order -> cos scores (E, n_way*Q, n_way), float64."""
    E = support.shape[0]
    P = H[support].reshape(E, n_way, k, H.shape[1]).mean(2)
    P = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
    return np.einsum("eqd,end->eqn", H[query], P).astype(np.float64)


def l_scores(prob, l_transform="log"):
    """View L scores from the dumped probabilities. Returns (scores float64, number of exact zeros replaced by float32 tiny)."""
    p = np.asarray(prob, dtype=np.float64)
    if l_transform == "prob":
        return p, 0
    if l_transform != "log":
        raise ValueError(l_transform)
    zero = p == 0
    return np.log(np.where(zero, TINY, p)), int(zero.sum())


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def mean_nll(logp, y):
    return float(-np.take_along_axis(logp, y[..., None], -1).mean())


def episode_acc(logp, y):
    return (logp.argmax(-1) == y).mean(-1)


def fit_temperatures(lL_valid, lD_valid, y_valid):
    """Grid search of T_L and s minimizing mean valid NLL (independently per view). Returns (T_L, s, iT, iS)."""
    nll_T = [mean_nll(log_softmax(lL_valid / t), y_valid) for t in TL_GRID]
    nll_s = [mean_nll(log_softmax(sv * lD_valid), y_valid) for sv in S_GRID]
    iT, iS = int(np.argmin(nll_T)), int(np.argmin(nll_s))
    return float(TL_GRID[iT]), float(S_GRID[iS]), iT, iS


def combine(lpL, lpD, alpha):
    return log_softmax(alpha * lpL + (1 - alpha) * lpD)


def select_alpha(lpL_valid, lpD_valid, y_valid, grid=A21):
    """argmin valid NLL over `grid`; ties -> the value closest to 1."""
    nll_a = [mean_nll(combine(lpL_valid, lpD_valid, a), y_valid) for a in grid]
    best = min(nll_a)
    return float(max(a for a, v in zip(grid, nll_a) if v == best))


def load_targets(run_dir):
    """Valid/test episodes of the run's best_valid_epoch from eval_logits.npz, ordered by ep_idx."""
    run = json.load(open(os.path.join(run_dir, "run.json")))
    Z = dict(np.load(os.path.join(run_dir, "eval_logits.npz")))
    e = run["final"]["best_epoch_valid"]
    k = Z["support"].shape[1] // N_WAY
    out = {"epoch": e, "k": k, "seed": run["config"]["seed"], "dataset": run["config"]["dataset"]}
    for mi, mode in ((0, "valid"), (1, "test")):
        m = np.where((Z["epoch"] == e) & (Z["mode"] == mi))[0]
        m = m[np.argsort(Z["ep_idx"][m])]
        out[mode] = {"ep_idx": Z["ep_idx"][m], "support": Z["support"][m], "query": Z["query"][m],
                     "classes": Z["classes"][m], "y": Z["query_y"][m].astype(int), "prob": Z["logits"][m]}
    return out


def prepare_run(run_dir, H, l_transform="log"):
    """Calibrated per-view log-probabilities for the run's target valid/test episodes."""
    t = load_targets(run_dir)
    lL = {}
    n_rep = 0
    for mode in ("valid", "test"):
        lL[mode], n = l_scores(t[mode]["prob"], l_transform)
        n_rep += n
    lD = {mode: view_d_scores(H, t[mode]["support"], t[mode]["query"], t["k"]) for mode in ("valid", "test")}
    T_L, s, iT, iS = fit_temperatures(lL["valid"], lD["valid"], t["valid"]["y"])
    lp = {mode: (log_softmax(lL[mode] / T_L), log_softmax(s * lD[mode])) for mode in ("valid", "test")}
    return {"targets": t, "lL": lL, "lD": lD, "lp": lp, "T_L": T_L, "s": s, "iT": iT, "iS": iS, "n_replaced": n_rep}


def evaluate(run_dir, H, l_transform="log", alpha="val"):
    """Per-episode test accuracy (best_valid_epoch, 50 episodes) of the combined prediction and the chosen (alpha, T_L, s)."""
    r = prepare_run(run_dir, H, l_transform)
    yv = r["targets"]["valid"]["y"]
    a = select_alpha(*r["lp"]["valid"], yv) if alpha == "val" else float(alpha)
    acc = episode_acc(combine(*r["lp"]["test"], a), r["targets"]["test"]["y"])
    return {"acc": acc, "alpha": a, "T_L": r["T_L"], "s": r["s"], "n_replaced": r["n_replaced"], "prepared": r}
