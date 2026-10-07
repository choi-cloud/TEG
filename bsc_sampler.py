"""T14 Blind-Spot Contrast (BSC) batch sampler helpers (no global RNG; callers pass np.random.default_rng(14000 + seed)).

- knn_graph: undirected cosine k-NN graph (CSR) over pool nodes, optionally restricted to groups (bsc_rwr).
- rwr_batch: random walk with restart on that graph until M distinct nodes; "jump" to a fresh uniform start node when the
  current node has no neighbour or 10*M consecutive steps add no new node.
- referee_neighbors: top-k cosine neighbours (row-normalized A_hat^2 X) inside the pool, computed once before training.
- referee_mask / pair_stats: batch-level negative mask and diagnostics (labels for diagnostics only).
"""
import numpy as np
import torch


def _topk_cos(Zq, Zk, k, exclude_self, block=4096):
    """Top-k cosine neighbours of rows of Zq among rows of Zk (both L2-normalized torch tensors on the same device).
    exclude_self: Zq is Zk (same order) and the diagonal is excluded. Returns a numpy int64 array [n, k'] (k' = min(k, available))."""
    n_k = Zk.shape[0]
    kk = min(k, n_k - 1 if exclude_self else n_k)
    if kk <= 0:
        return np.zeros((Zq.shape[0], 0), dtype=np.int64)
    out = []
    for s in range(0, Zq.shape[0], block):
        sim = Zq[s : s + block] @ Zk.T
        if exclude_self:
            r = torch.arange(sim.shape[0], device=sim.device)
            sim[r, r + s] = float("-inf")
        out.append(sim.topk(kk, dim=1).indices.cpu().numpy())
    return np.concatenate(out).astype(np.int64)


def knn_graph(Z, k, groups=None, block=4096):
    """Z: [n, d] torch (pool nodes, any scale; L2-normalized here). groups: None (whole pool) or int numpy [n].
    Returns (indptr, indices, stats) of the undirected k-NN graph in pool-local indices."""
    Z = torch.nn.functional.normalize(Z, dim=1)
    n = Z.shape[0]
    src, dst = [], []
    if groups is None:
        nb = _topk_cos(Z, Z, k, True, block)
        src.append(np.repeat(np.arange(n), nb.shape[1]))
        dst.append(nb.reshape(-1))
        sizes = np.array([n])
    else:
        gids = np.unique(groups)
        sizes = []
        for g in gids:
            members = np.where(groups == g)[0]
            sizes.append(len(members))
            Zg = Z[torch.from_numpy(members).to(Z.device)]
            nb = _topk_cos(Zg, Zg, k, True, block)
            src.append(np.repeat(members, nb.shape[1]))
            dst.append(members[nb.reshape(-1)])
        sizes = np.array(sizes)
    src, dst = np.concatenate(src), np.concatenate(dst)
    a, b = np.concatenate([src, dst]), np.concatenate([dst, src])
    key = np.unique(a * n + b)
    a, b = key // n, key % n
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    indptr = np.cumsum(indptr)
    deg = np.diff(indptr)
    stats = {"n_groups": int(len(sizes)), "group_size_median": float(np.median(sizes)), "n_isolated": int((deg == 0).sum()),
             "n_edges_undirected": int(len(key) // 2)}
    return indptr, b.astype(np.int64), stats


def rwr_batch(indptr, indices, m, alpha, rng, chunk=4096, return_starts=False):
    """Random walk with restart on the CSR graph (pool-local ids). Returns (batch [m] int64 in visit order, n_jumps);
    return_starts=True also returns the batch positions where a segment (first start or jump) begins (T14 R3 check only)."""
    n = len(indptr) - 1
    m = min(m, n)
    in_batch = np.zeros(n, dtype=bool)
    batch = []
    jumps = 0
    start = int(rng.integers(n))
    cur = start
    in_batch[cur] = True
    batch.append(cur)
    starts = [0]
    stale = 0
    u_restart, u_move, pos = None, None, chunk
    while len(batch) < m:
        if pos >= chunk:
            u_restart, u_move, pos = rng.random(chunk), rng.random(chunk), 0
        lo, hi = indptr[cur], indptr[cur + 1]
        if hi == lo or stale >= 10 * m:
            # jump: fresh uniform start among pool nodes not yet in the batch
            free = np.flatnonzero(~in_batch)
            start = int(free[rng.integers(len(free))])
            cur = start
            jumps += 1
            stale = 0
            in_batch[cur] = True
            starts.append(len(batch))
            batch.append(cur)
            continue
        if u_restart[pos] < alpha:
            cur = start
        else:
            cur = int(indices[lo + int(u_move[pos] * (hi - lo))])
        pos += 1
        if in_batch[cur]:
            stale += 1
        else:
            in_batch[cur] = True
            batch.append(cur)
            stale = 0
    if return_starts:
        return np.array(batch, dtype=np.int64), jumps, np.array(starts, dtype=np.int64)
    return np.array(batch, dtype=np.int64), jumps


def referee_neighbors(H, k, block=4096):
    """H: [n, d] torch (row-normalized A_hat^2 X of the pool). Returns pool-local top-k cosine neighbours [n, k]."""
    H = torch.nn.functional.normalize(H, dim=1)
    return _topk_cos(H, H, k, True, block)


def referee_mask(batch_local, ref_nb, n_pool, device):
    """[m, m] bool: (i, j), i != j, masked if j in R(i) or i in R(j)."""
    m = len(batch_local)
    pos = np.full(n_pool, -1, dtype=np.int64)
    pos[batch_local] = np.arange(m)
    nb = pos[ref_nb[batch_local]]  # [m, k] batch positions or -1
    rows = np.repeat(np.arange(m), nb.shape[1])
    cols = nb.reshape(-1)
    keep = cols >= 0
    mask = torch.zeros(m, m, dtype=torch.bool, device=device)
    mask[torch.from_numpy(rows[keep]).to(device), torch.from_numpy(cols[keep]).to(device)] = True
    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


def pair_stats(labels, groups, mask):
    """labels, groups: [m] torch (diagnostics only); mask: [m, m] bool or None. Over unordered pairs i < j:
    fn_rate = same-label share among pairs used as negatives (not masked), same_proto_rate = same-group share (all pairs),
    masked_frac = masked share."""
    m = len(labels)
    iu = torch.triu_indices(m, m, 1, device=labels.device)
    same_y = labels[iu[0]] == labels[iu[1]]
    same_g = groups[iu[0]] == groups[iu[1]]
    used = ~mask[iu[0], iu[1]] if mask is not None else torch.ones_like(same_y)
    n_used = int(used.sum())
    return {"fn_rate": float((same_y & used).sum()) / max(n_used, 1), "same_proto_rate": float(same_g.float().mean()),
            "masked_frac": 1.0 - n_used / len(same_y)}
