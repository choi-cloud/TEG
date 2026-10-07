"""T14 R3: RWR unit checks (bsc_sampler.rwr_batch). Generators np.random.default_rng(14000 + s), s = 0..19.
(a) [replaced, instruction addendum 2026-10-08] ring lattice, 2,000 nodes, 5 neighbours on each side (degree 10): batch split
    into segments at jumps; per node the ring distance to its segment's start node, averaged over nodes; pass if for M in
    {20, 128} mean distance(alpha = 0.9) < mean distance(alpha = 0.05) (mean over the 20 generators).
(a0) [record only] path graph, 2,000 nodes: batch node-index std for alpha = 0.9 vs 0.05 (M in 5, 10, 20, 50, 100, 1024), with jumps.
(b) graph whose groups all have size 1 (no edges): the batch is filled with M nodes by jumps."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bsc_sampler as b  # noqa: E402

n = 2000
a = np.concatenate([np.arange(n - 1), np.arange(1, n)])
c = np.concatenate([np.arange(1, n), np.arange(n - 1)])
o = np.argsort(a * n + c, kind="stable")
a, c = a[o], c[o]
indptr = np.cumsum(np.concatenate([[0], np.bincount(a, minlength=n)])).astype(np.int64)
# (a) ring lattice
src = np.repeat(np.arange(n), 10)
dst = (src + np.tile(np.concatenate([np.arange(1, 6), -np.arange(1, 6)]), n)) % n
o = np.argsort(src * n + dst, kind="stable")
r_ip = np.cumsum(np.concatenate([[0], np.bincount(src, minlength=n)])).astype(np.int64)
r_ix = dst[o].astype(np.int64)
ok_ring = True
print("(a) ring lattice n=2000 k=5+5: M | alpha | mean ring distance to segment start | mean jumps (20 generators)")
for M in (20, 128):
    res = {}
    for al in (0.9, 0.05):
        ds, js = [], []
        for s in range(20):
            bt, j, st = b.rwr_batch(r_ip, r_ix, M, al, np.random.default_rng(14000 + s), return_starts=True)
            seg_start = bt[st][np.searchsorted(st, np.arange(M), side="right") - 1]
            d = np.abs(bt - seg_start)
            ds.append(np.minimum(d, n - d).mean())
            js.append(j)
        res[al] = (np.mean(ds), np.mean(js))
        print(f"{M} | {al} | {res[al][0]:.3f} | {res[al][1]:.2f}")
    ok_ring &= res[0.9][0] < res[0.05][0]
print(f"(a) R3(a) pass (distance alpha=0.9 < alpha=0.05 for M in 20, 128): {ok_ring}")
ok_a = True
print("(a0) path graph n=2000: M | alpha | mean std of batch indices | mean jumps (20 generators)")
for M in (5, 10, 20, 50, 100, 1024):
    res = {}
    for al in (0.9, 0.05):
        st, js = [], []
        for s in range(20):
            bt, j = b.rwr_batch(indptr, c, M, al, np.random.default_rng(14000 + s))
            assert len(bt) == M and len(set(bt.tolist())) == M
            st.append(bt.std())
            js.append(j)
        res[al] = (np.mean(st), np.mean(js))
        print(f"{M} | {al} | {res[al][0]:.2f} | {res[al][1]:.2f}")
    ok_a &= res[0.9][0] < res[0.05][0]
print(f"(a0) [record only] std(alpha=0.9) < std(alpha=0.05) for every M: {ok_a}")
# (b) all groups of size 1 -> knn_graph with groups gives no edges
import torch  # noqa: E402
Z = torch.randn(300, 8, generator=torch.Generator().manual_seed(14000))
ip, ix, stt = b.knn_graph(Z, 10, groups=np.arange(300))
bt, j = b.rwr_batch(ip, ix, 128, 0.2, np.random.default_rng(14000))
ok_b = len(bt) == 128 and len(set(bt.tolist())) == 128 and stt["n_isolated"] == 300
print(f"(b) size-1 groups: graph stats {stt}; batch {len(bt)} distinct {len(set(bt.tolist()))} jumps {j}: {ok_b}")
