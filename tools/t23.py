"""T23 queue (analysis only, no training): multi-resolution cluster structure vs labels.

Subcommands
  python tools/t23.py a-run <ds> <k>      T23a: cluster assignments for cell (ds, k) -> results/T23/clusters/...
  python tools/t23.py a-gates             T23a: Gc (synthetic ARI) -> results/T23/gates_a.json
  python tools/t23.py a-report            T23a: Ge / Gf, tables A1-A3 -> reports/T23a_summary.md
  python tools/t23.py b-gates             T23b: Hb, Hc, Hd
  python tools/t23.py b-run <ds> <k>      T23b: B-1, B-2 -> results/T23/T23b/<ds>_5w<k>s/seed<s>.csv
  python tools/t23.py b-report            T23b: tables B1-B6, R1-R7, P116-P120
  python tools/t23.py c-gates / c-run <ds> <k> / c-report     T23c

Views: X (utils.load_data), AX = row-normalized A_hat X, A2X = row-normalized A_hat^2 X (analyze_t19.fixed_views),
rand1 (analyze_t19.rand1_views, offset 2000), TEGe = T22 base emb_best.npy, CONe = T22 infonce_all (selected lambda) emb_best.npy.
All views are row L2-normalized; cosine similarity. Dedicated RNG: torch.Generator 17000 + s (k-means++), numpy
default_rng 17100 + s / 17200 + s (T23c episodes), default_rng 17300 + s (SHUF).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)
R22 = os.path.join(ROOT, "results", "T22")
OUT = os.path.join(ROOT, "results", "T23")
CL = os.path.join(OUT, "clusters")
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
RES = [16, 64, 256, 1024, 4096]
GROUPS = ["base", "valid", "test"]
LAM_ALL = {("Amazon_clothing", 1): 10, ("Amazon_clothing", 5): 10, ("dblp", 1): 10, ("dblp", 5): 10,
           ("Amazon_electronics", 1): 100, ("Amazon_electronics", 5): 30}
FIXED = ["X", "AX", "A2X"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
MAX_IT, TOL = 50, 1e-3


def emb_dir(E, ds, k, s):
    cfg = "base" if E == "TEGe" else f"infonce_all_l{LAM_ALL[(ds, k)]}"
    return os.path.join(R22, cfg, f"{ds}_5w{k}s", f"seed{s}")


def cl_dir(view, ds, k, s):
    if view in FIXED:
        return os.path.join(CL, view, ds, "seed0")
    if view == "rand1":
        return os.path.join(CL, view, ds, f"seed{s}")
    return os.path.join(CL, view, f"{ds}_5w{k}s", f"seed{s}")


# ------------------------------------------------------------------ spherical k-means (T23a §3)
def l2n_t(Z):
    t = torch.as_tensor(np.asarray(Z), dtype=torch.float32, device=DEV)
    return torch.nn.functional.normalize(t, dim=1)


def kmeanspp(Zn, k, g):
    """k-means++ with distance 1 - cos, weights D^2; first centre torch.randint(N, generator=g), then one uniform from g per
    centre mapped through the cumulative D^2 (inverse CDF, searchsorted right)."""
    N = Zn.shape[0]
    first = int(torch.randint(N, (1,), generator=g))
    idx = [first]
    D = (1.0 - Zn @ Zn[first]).clamp_min(0)
    for _ in range(1, k):
        # inverse-CDF draw on the GPU: one uniform from the CPU generator g per centre
        cw = torch.cumsum(D.double() ** 2, 0)
        r = float(torch.rand(1, generator=g, dtype=torch.float64)) * float(cw[-1])
        if float(cw[-1]) <= 0:
            nxt = int(np.setdiff1d(np.arange(N), idx)[0])
        else:
            nxt = min(int(torch.searchsorted(cw, torch.tensor([r], dtype=torch.float64, device=cw.device), right=True)), N - 1)
        idx.append(nxt)
        D = torch.minimum(D, (1.0 - Zn @ Zn[nxt]).clamp_min(0))
    return Zn[torch.as_tensor(idx, device=Zn.device)].clone()


def assign_of(Zn, C, block=8192):
    out, best = [], []
    for s in range(0, Zn.shape[0], block):
        v, i = (Zn[s : s + block] @ C.T).max(1)
        out.append(i)
        best.append(v)
    return torch.cat(out), torch.cat(best)


def fix_empty(a, best, C, Zn, k):
    """Re-seed empty clusters with the nodes of lowest cosine to their own centre (ties -> smaller index). Deterministic."""
    cnt = torch.bincount(a, minlength=k)
    empty = torch.where(cnt == 0)[0].cpu().numpy()
    if len(empty) == 0:
        return a, 0
    b = best.double().cpu().numpy()
    order = np.lexsort((np.arange(len(b)), b))
    a = a.clone()
    used = 0
    for e in empty:
        while True:
            n = int(order[used])
            used += 1
            if int(torch.bincount(a, minlength=k)[a[n]]) > 1:
                break
        a[n] = int(e)
        C[int(e)] = Zn[n]
    return a, len(empty)


def skmeans(Zn, k, g):
    t0 = time.time()
    C = kmeanspp(Zn, k, g)
    prev, n_empty, it, frac = None, 0, 0, 1.0
    for it in range(1, MAX_IT + 1):
        a, best = assign_of(Zn, C)
        a, ne = fix_empty(a, best, C, Zn, k)
        n_empty += ne
        frac = 1.0 if prev is None else float((a != prev).float().mean())
        C = torch.zeros_like(C).index_add_(0, a, Zn)
        C = torch.nn.functional.normalize(C, dim=1)
        prev = a
        if frac < TOL:
            break
    return a.cpu().numpy().astype(np.int32), C, {"k": k, "iters": it, "last_changed": frac, "empty_reseeded": n_empty,
                                                  "n_clusters": int(len(np.unique(a.cpu().numpy()))), "sec": time.time() - t0}


def hier_from(a5, C5):
    """H family: Ward linkage on the 4096 centres, fcluster maxclust k in (1024, 256, 64, 16)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    t0 = time.time()
    Lk = linkage(C5.double().cpu().numpy(), method="ward")
    H = np.zeros((len(a5), 5), dtype=np.int32)
    H[:, 4] = a5
    got = {}
    for lv, k in zip(range(4), RES[:4]):
        lab = fcluster(Lk, t=k, criterion="maxclust").astype(np.int32) - 1
        H[:, lv] = lab[a5]
        got[k] = int(len(np.unique(H[:, lv])))
    return H, {"n_clusters": got, "sec": time.time() - t0}


def cluster_view(Z, seed, d_out):
    Zn = l2n_t(Z)
    g = torch.Generator().manual_seed(17000 + seed)
    M = np.zeros((Zn.shape[0], 5), dtype=np.int32)
    meta = {"M": [], "seed_generator": 17000 + seed}
    C5 = None
    for lv, k in enumerate(RES):
        a, C, m = skmeans(Zn, k, g)
        M[:, lv] = a
        meta["M"].append(m)
        C5 = C
    H, mh = hier_from(M[:, 4], C5)
    meta["H"] = mh
    os.makedirs(d_out, exist_ok=True)
    np.save(os.path.join(d_out, "assign_M.npy"), M)
    np.save(os.path.join(d_out, "assign_H.npy"), H)
    json.dump(meta, open(os.path.join(d_out, "meta.json"), "w"), indent=1)
    del Zn
    torch.cuda.empty_cache()
    return M, H, meta


def gate_split_t22(ds, s):
    """Ga: T22 base / infonce_all run labels.npy, split.json vs utils.load_data after seed_everything(s) (analyze_t19.gate_split way)."""
    from utils import load_data, seed_everything

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        seed_everything(s)
        out = load_data(ds)
    finally:
        os.chdir(cwd)
    labels, tr, va, te = out[3].numpy(), out[5], out[6], out[7]
    res = []
    for E in ("TEGe", "CONe"):
        for k in SHOTS:
            d = emb_dir(E, ds, k, s)
            sp = json.load(open(os.path.join(d, "split.json")))
            lab = np.load(os.path.join(d, "labels.npy"))
            res.append({"E": E, "shot": k, "seed": s, "labels_equal": bool(np.array_equal(lab, labels)),
                        "lists_equal": all([int(x) for x in sp[key]] == [int(x) for x in ref] for key, ref in
                                           (("class_list_train", tr), ("class_list_valid", va), ("class_list_test", te)))})
    return res


def a_run(ds, k):
    t_all = time.time()
    log = {"cell": f"{ds}_5w{k}s", "views": {}, "gb": [], "ga": []}
    if k == 1:
        import analyze_t19 as a19

        t0 = time.time()
        fv = a19.fixed_views(ds)
        log["t_fixed_views"] = time.time() - t0
        for v in FIXED:
            t0 = time.time()
            M, H, meta = cluster_view(fv[v], 0, cl_dir(v, ds, k, 0))
            log["views"][f"{v}|0"] = time.time() - t0
            if v == "AX" and ds == "Amazon_clothing":  # Gd: rerun the M chain up to k = 256
                Zn = l2n_t(fv[v])
                g = torch.Generator().manual_seed(17000)
                for lv, kk in enumerate(RES[:3]):
                    a2, _, _ = skmeans(Zn, kk, g)
                log["Gd_equal"] = bool(np.array_equal(a2, M[:, 2]))
                del Zn
        del fv
        t0 = time.time()
        r1, t_ds = a19.rand1_views(ds)
        log["t_rand1_views"] = time.time() - t0
        for s in SEEDS:
            t0 = time.time()
            cluster_view(r1[s], s, cl_dir("rand1", ds, k, s))
            log["views"][f"rand1|{s}"] = time.time() - t0
        del r1
        for s in SEEDS:
            log["ga"] += gate_split_t22(ds, s)
    for E in ("TEGe", "CONe"):
        for s in SEEDS:
            p = os.path.join(emb_dir(E, ds, k, s), "emb_best.npy")
            if not os.path.exists(p):
                log["gb"].append({"E": E, "seed": s, "exists": False})
                continue
            Z = np.load(p)
            log["gb"].append({"E": E, "seed": s, "exists": True, "shape": list(Z.shape)})
            t0 = time.time()
            cluster_view(Z, s, cl_dir(E, ds, k, s))
            log["views"][f"{E}|{s}"] = time.time() - t0
    log["sec"] = time.time() - t_all
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    json.dump(log, open(os.path.join(OUT, "logs", f"a_{ds}_5w{k}s.json"), "w"), indent=1)
    print(json.dumps({k2: v for k2, v in log.items() if k2 != "ga"}), flush=True)


def synth_blobs(g, n=200, d=16, c=3):
    centers = torch.nn.functional.normalize(torch.randn(c, d, generator=g), dim=1) * 8.0
    X = torch.cat([centers[i] + torch.randn(n, d, generator=g) for i in range(c)])
    return X.numpy(), np.repeat(np.arange(c), n)


def a_gates():
    from sklearn.metrics import adjusted_rand_score

    g = torch.Generator().manual_seed(17000)
    X, y = synth_blobs(g)
    Zn = l2n_t(X)
    a, _, m = skmeans(Zn, 3, torch.Generator().manual_seed(17000))
    out = {"Gc_ari": float(adjusted_rand_score(y, a.astype(int))), "Gc_meta": m}
    json.dump(out, open(os.path.join(OUT, "gates_a.json"), "w"), indent=1)
    print(out)


def all_assign_dirs():
    out = []
    for ds in DATASETS:
        for v in FIXED:
            out.append((v, ds, None, 0))
        for s in SEEDS:
            out.append(("rand1", ds, None, s))
        for k in SHOTS:
            for E in ("TEGe", "CONe"):
                for s in SEEDS:
                    out.append((E, ds, k, s))
    return out


def nested_ok(H):
    for l1 in range(5):
        for l2 in range(l1 + 1, 5):
            pair = np.unique(np.stack([H[:, l2], H[:, l1]], 1), axis=0)
            if len(np.unique(pair[:, 0])) != len(pair):
                return False
    return True


def size_stats(a):
    _, cnt = np.unique(a, return_counts=True)
    return {"n": int(len(cnt)), "min": int(cnt.min()), "med": float(np.median(cnt)), "max": int(cnt.max()),
            "n_single": int((cnt == 1).sum()), "frac_largest": float(cnt.max() / len(a))}


def a_report():
    rows, Ge, Gf_M, Gf_H = {}, [], [], []
    metas = {}
    for v, ds, k, s in all_assign_dirs():
        d = cl_dir(v, ds, k, s)
        M, H = np.load(os.path.join(d, "assign_M.npy")), np.load(os.path.join(d, "assign_H.npy"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        key = (v, ds, k, s)
        metas[key] = meta
        if not nested_ok(H):
            Ge.append(key)
        for lv, kk in enumerate(RES):
            sm, sh = size_stats(M[:, lv]), size_stats(H[:, lv])
            rows[key + ("M", kk)] = sm
            rows[key + ("H", kk)] = sh
            if sm["n"] != kk:
                Gf_M.append((key, kk, sm["n"]))
            if sh["n"] != kk:
                Gf_H.append((key, kk, sh["n"]))
    logs = {f"{ds}_5w{k}s": json.load(open(os.path.join(OUT, "logs", f"a_{ds}_5w{k}s.json"))) for ds in DATASETS for k in SHOTS}
    gates = json.load(open(os.path.join(OUT, "gates_a.json")))
    ga = [r for lg in logs.values() for r in lg["ga"]]
    gb = [r for lg in logs.values() for r in lg["gb"]]
    G = {"Ga": all(r["labels_equal"] and r["lists_equal"] for r in ga) and len(ga) == 36, "Ga_n": len(ga),
         "Gb": all(r["exists"] and r["shape"][1] == 64 for r in gb) and len(gb) == 36, "Gb_n": len(gb),
         "Gc_ari": gates["Gc_ari"], "Gd": logs["Amazon_clothing_5w1s"].get("Gd_equal"), "Ge_fail": [list(map(str, x)) for x in Ge],
         "Gf_M": [str(x) for x in Gf_M], "Gf_H": [str(x) for x in Gf_H], "n_assign": len(metas)}
    G["pass"] = G["Ga"] and G["Gb"] and G["Gc_ari"] == 1.0 and G["Gd"] is True and not Ge
    json.dump(G, open(os.path.join(OUT, "gate_checks_a.json"), "w"), indent=1)

    L = ["# T23a 요약 — 다중 해상도 클러스터 (학습 없음)", ""]
    L.append("- 뷰: X, AX, A2X(데이터셋당 1, seed0 생성기), rand1(seed 0–2), TEGe·CONe((데이터셋, shot) × seed 0–2). 해상도 16, 64, 256, 1024, 4096. "
             "M = 해상도별 독립 구면 k-means, H = M 4096 중심의 Ward 병합. 크기 통계는 rand1·TEGe·CONe에서 seed 평균.")
    L.append("")
    L.append("## A1 — 클러스터 크기 (실제 수 / 최소 / 중앙값 / 최대 / 크기 1 수 / 최대 비율)")
    L.append("| 데이터셋 | 뷰 | 셀 | 계열 | k | 실제 수 | 최소 | 중앙값 | 최대 | 크기 1 | 최대 비율 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        groups = [(v, None, [0]) for v in FIXED] + [("rand1", None, SEEDS)] + [(E, k, SEEDS) for E in ("TEGe", "CONe") for k in SHOTS]
        for v, k, ss in groups:
            for F in ("M", "H"):
                for kk in RES:
                    st = [rows[(v, ds, k, s, F, kk)] for s in ss]
                    mm = lambda key: np.mean([x[key] for x in st])
                    L.append(f"| {ds} | {v} | {'—' if k is None else f'{k}-shot'} | {F} | {kk} | {mm('n'):.1f} | {mm('min'):.1f} | {mm('med'):.1f} | "
                             f"{mm('max'):.1f} | {mm('n_single'):.1f} | {mm('frac_largest'):.4f} |")
    L.append("")
    L.append("## A2 — k-means 반복 수와 마지막 바뀐 비율 (해상도별, 뷰 종류 내 최댓값)")
    L.append("| 데이터셋 | 뷰 | " + " | ".join(f"k={kk} 반복 / 바뀐 비율 / 재심기" for kk in RES) + " |")
    L.append("|---|---|" + "---|" * len(RES))
    for ds in DATASETS:
        for v in FIXED + ["rand1", "TEGe", "CONe"]:
            ms = [m for key, m in metas.items() if key[0] == v and key[1] == ds]
            cols = []
            for lv in range(5):
                cols.append(f"{max(m['M'][lv]['iters'] for m in ms)} / {max(m['M'][lv]['last_changed'] for m in ms):.5f} / "
                            f"{max(m['M'][lv]['empty_reseeded'] for m in ms)}")
            L.append(f"| {ds} | {v} | " + " | ".join(cols) + " |")
    L.append("")
    L.append("## A3 — 실행 시간 (초, 클러스터링 1회 = M 5개 + H)")
    L.append("| 셀 | 뷰 | 시간 |")
    L.append("|---|---|---|")
    for c, lg in logs.items():
        for vk, t in lg["views"].items():
            L.append(f"| {c} | {vk} | {t:.1f} |")
        L.append(f"| {c} | 프로세스 전체 | {lg['sec']:.1f} |")
    L.append("")
    open(os.path.join(ROOT, "reports", "T23a_summary.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(G))


# ------------------------------------------------------------------ T23b
MARG = 16


def labels_eff(lab, gid):
    """Nodes in no class list (gid -1) get a unique negative label (counted as another class)."""
    y = lab.astype(np.int64).copy()
    y[gid < 0] = -1 - np.where(gid < 0)[0]
    return y


def restricted_knn(Zn, anchors, cl=None, K=10, block=2048, sim_cache=None):
    """Top-K cosine neighbours of each anchor among all nodes (cl None) or among nodes in the anchor's cluster (cl: torch int
    tensor [N] on DEV), self excluded. Tie rule as analyze_t19.knn_sorted (sim desc, node index asc). Returns (nb [n, K] global
    ids, -1 where missing; n_found [n])."""
    N = Zn.shape[0]
    out_nb, out_n = [], []
    for s in range(0, len(anchors), block):
        a = np.asarray(anchors[s : s + block])
        at = torch.as_tensor(a, device=Zn.device)
        sim = Zn[at] @ Zn.T if sim_cache is None else sim_cache[s // block]
        sim = sim.clone()
        sim[torch.arange(len(a), device=sim.device), at] = float("-inf")
        if cl is not None:
            sim.masked_fill_(cl[at][:, None] != cl[None, :], float("-inf"))
        m = min(K + MARG, N - 1)
        v, i = sim.topk(m, dim=1)
        v, i = v.cpu().numpy(), i.cpu().numpy()
        order = np.lexsort((i, -v), axis=1)
        v, i = np.take_along_axis(v, order, 1), np.take_along_axis(i, order, 1)
        last = v[:, -1].copy()
        v, i = v[:, :K].copy(), i[:, :K].copy()
        rows = np.where((last == v[:, -1]) & np.isfinite(last))[0]
        for r in rows:  # ties past the fetched candidates: exact re-order over the row
            full = sim[r].cpu().numpy()
            o = np.lexsort((np.arange(N), -full))[:K]
            v[r], i[r] = full[o], o
        fin = np.isfinite(v)
        i[~fin] = -1
        out_nb.append(i)
        out_n.append(fin.sum(1))
    return np.concatenate(out_nb), np.concatenate(out_n)


def pur_rows(nb, n, ya, y, K):
    """Per-anchor same-label fraction over the first min(K, n) neighbours; NaN where n == 0."""
    nbK = nb[:, :K]
    same = (y[np.where(nbK >= 0, nbK, 0)] == ya[:, None]) & (nbK >= 0)
    cnt = np.minimum(n, K)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, same.sum(1) / cnt, np.nan), cnt < K


def b1_purity(assign_col, y):
    """p(u) = same-label others in the cluster / (cluster size - 1); NaN for singletons."""
    key = assign_col.astype(np.int64) * (y.max() + 1 - y.min() + 1) + (y - y.min())
    _, inv_k, cnt_k = np.unique(key, return_inverse=True, return_counts=True)
    _, inv_c, cnt_c = np.unique(assign_col, return_inverse=True, return_counts=True)
    same, size = cnt_k[inv_k] - 1, cnt_c[inv_c] - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(size > 0, same / np.maximum(size, 1), np.nan)


def load_assign(view, ds, k, s, F):
    return np.load(os.path.join(cl_dir(view, ds, k, s), f"assign_{F}.npy"))


def shuf_of(A, s):
    rng = np.random.default_rng(17300 + s)
    perms = [rng.permutation(A.shape[0]) for _ in range(5)]
    return np.stack([A[perms[lv], lv] for lv in range(5)], 1), perms


def b_run(ds, k):
    import csv

    import analyze_t19 as a19

    t_all = time.time()
    od = os.path.join(OUT, "T23b", f"{ds}_5w{k}s")
    os.makedirs(od, exist_ok=True)
    timing = {}
    for s in SEEDS:
        t0 = time.time()
        sp, lab = a19.split_of(emb_dir("TEGe", ds, k, s))
        gid = a19.group_ids(sp, lab)
        y = labels_eff(lab, gid)
        N = len(y)
        anchors = np.where(gid >= 0)[0]
        gof = gid[anchors]
        A = {F: {v: load_assign(v, ds, k, s, F) for v in FIXED + ["rand1", "TEGe", "CONe"]} for F in ("H", "M")}
        for F in ("H", "M"):
            A[F]["SHUF"] = shuf_of(A[F]["AX"], s)[0]
        rows = []
        # B-2 (order: TEGe H, CONe H, then M; K = 5 comes from the same sorted lists)
        for E in ("TEGe", "CONe"):
            Zn = a19.normed(np.load(os.path.join(emb_dir(E, ds, k, s), "emb_best.npy")))
            nb_u, n_u = restricted_knn(Zn, anchors, None, K=50)
            u = {K: pur_rows(nb_u, n_u, y[anchors], y, K)[0] for K in (5, 10, 20, 50)}
            for F in ("H", "M"):
                for V in ["X", "AX", "A2X", "rand1", "SELF", "SHUF"]:
                    AA = A[F][E if V == "SELF" else V]
                    for lv in range(5):
                        cl = torch.as_tensor(AA[:, lv].astype(np.int64), device=DEV)
                        nb, n = restricted_knn(Zn, anchors, cl, K=10)
                        for K in (5, 10):
                            r, short = pur_rows(nb, n, y[anchors], y, K)
                            for gi, g in enumerate(GROUPS):
                                m = (gof == gi) & np.isfinite(r)
                                d = r[m] - u[K][m]
                                rows.append({"part": "B2", "E": E, "V": V, "F": F, "l": lv + 1, "K": K, "group": g, "r": float(r[m].mean()),
                                             "u": float(u[K][m].mean()), "gain": float(d.mean()), "gain_se_anchor": float(d.std(ddof=1) / math.sqrt(len(d))),
                                             "n_anchor": int(m.sum()), "n_excluded": int(((gof == gi) & ~np.isfinite(r)).sum()),
                                             "frac_short": float(short[(gof == gi)].mean())})
            for K in (5, 10, 20, 50):
                for gi, g in enumerate(GROUPS):
                    rows.append({"part": "U", "E": E, "K": K, "group": g, "u": float(u[K][gof == gi].mean())})
            del Zn
            torch.cuda.empty_cache()
        # B-1
        for F in ("H", "M"):
            for V in FIXED + ["rand1", "TEGe", "CONe"]:
                for lv in range(5):
                    p = b1_purity(A[F][V][:, lv], y)[anchors]
                    for gi, g in enumerate(GROUPS):
                        m = gof == gi
                        rows.append({"part": "B1", "V": V, "F": F, "l": lv + 1, "group": g, "p": float(np.nanmean(p[m])),
                                     "frac_single": float(np.isnan(p[m]).mean())})
        keys = sorted({kk for r in rows for kk in r})
        with open(os.path.join(od, f"seed{s}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        timing[s] = time.time() - t0
        print(ds, k, s, f"{timing[s]:.1f}s", flush=True)
    json.dump({"sec": time.time() - t_all, "seed_sec": timing}, open(os.path.join(od, "timing.json"), "w"))


def synth_tie(n=200, d=16):
    """Deterministic synthetic data (generator 17000) with duplicated rows (exact ties), labels i % 5, clusters per level."""
    g = torch.Generator().manual_seed(17000)
    Z = torch.randn(n, d, generator=g).double()
    Z[n // 2 : n // 2 + 20] = Z[:20]
    lab = np.arange(n) % 5
    A = np.stack([np.arange(n) % c for c in (2, 3, 7, 13, 40)], 1).astype(np.int32)
    return Z, lab, A


def b_gates():
    import analyze_t19 as a19

    out = {}
    # Hb: u_10 (TEGe, Amazon_clothing 1-shot seed 0, test) vs analyze_t19.measure_view pur_all
    ds, k, s = "Amazon_clothing", 1, 0
    sp, lab = a19.split_of(emb_dir("TEGe", ds, k, s))
    gid = a19.group_ids(sp, lab)
    y = labels_eff(lab, gid)
    Z = np.load(os.path.join(emb_dir("TEGe", ds, k, s), "emb_best.npy"))
    ref = a19.measure_view(Z, lab, gid)["test"][10]["pur_all"]
    Zn = a19.normed(Z)
    anchors = np.where(gid == 2)[0]
    nb, n = restricted_knn(Zn, anchors, None, K=10)
    u10 = float(pur_rows(nb, n, y[anchors], y, 10)[0].mean())
    out["Hb"] = {"u10": u10, "a19_pur_all": ref, "absdiff": abs(u10 - ref)}
    # Hc(a): one cluster for all nodes
    nb1, n1 = restricted_knn(Zn, anchors, torch.zeros(len(y), dtype=torch.int64, device=DEV), K=10)
    r10 = float(pur_rows(nb1, n1, y[anchors], y, 10)[0].mean())
    out["Hc_a"] = {"r10": r10, "u10": u10, "absdiff": abs(r10 - u10), "nb_equal": bool(np.array_equal(nb1, nb))}
    # Hc(b): synthetic, restricted kNN vs brute force per anchor
    Zs, ls, As = synth_tie()
    Zsn = torch.nn.functional.normalize(Zs, dim=1).to(DEV)
    S = (Zsn @ Zsn.T).cpu().numpy()
    bad, n_short = 0, 0
    for lv in range(5):
        for K in (5, 10):
            nbs, ns = restricted_knn(Zsn, np.arange(200), torch.as_tensor(As[:, lv].astype(np.int64), device=DEV), K=K, block=64)
            for a in range(200):
                cand = np.where((As[:, lv] == As[a, lv]) & (np.arange(200) != a))[0]
                o = cand[np.lexsort((cand, -S[a, cand]))][:K]
                got = nbs[a][: ns[a]]
                bad += int(not np.array_equal(o, got))
                n_short += int(len(cand) < K)
    out["Hc_b"] = {"mismatch_rows": bad, "rows": 200 * 10, "short_rows": n_short}
    # Hd: B-1 vs brute force
    badp = 0
    for lv in range(5):
        p = b1_purity(As[:, lv], ls.astype(np.int64))
        for a in range(200):
            mem = np.where((As[:, lv] == As[a, lv]) & (np.arange(200) != a))[0]
            q = np.nan if len(mem) == 0 else (ls[mem] == ls[a]).mean()
            badp += int(not ((np.isnan(q) and np.isnan(p[a])) or q == p[a]))
    out["Hd"] = {"mismatch": badp, "rows": 1000}
    out["pass"] = out["Hb"]["absdiff"] <= 1e-9 and out["Hc_a"]["absdiff"] <= 1e-9 and bad == 0 and badp == 0
    json.dump(out, open(os.path.join(OUT, "gates_b.json"), "w"), indent=1)
    print(json.dumps(out))


CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]


def seed_stats(xs):
    xs = np.asarray(xs, dtype=float)
    m = float(xs.mean())
    se = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
    sg = np.sign(m)
    agree = int(np.sum(np.sign(xs) == sg)) if m != 0 else 0
    return m, se, agree


def clear_pos(xs):
    m, se, ag = seed_stats(xs)
    return m > 0 and m > 2 * se and ag == len(xs)


def clear_neg(xs):
    m, se, ag = seed_stats(xs)
    return m < 0 and -m > 2 * se and ag == len(xs)


def fmt_ms(xs, f="{:+.4f}"):
    m, se, ag = seed_stats(xs)
    return f"{f.format(m)} ± {se:.4f} ({ag}/{len(xs)})"


def load_b():
    import csv

    D = {}
    for ds, k in CELLS:
        for s in SEEDS:
            p = os.path.join(OUT, "T23b", f"{ds}_5w{k}s", f"seed{s}.csv")
            if os.path.exists(p):
                D[(ds, k, s)] = list(csv.DictReader(open(p)))
    return D


def b_index(D):
    G, U, B1 = {}, {}, {}
    for (ds, k, s), rows in D.items():
        for r in rows:
            if r["part"] == "B2":
                G[(ds, k, s, r["E"], r["V"], r["F"], int(r["l"]), int(r["K"]), r["group"])] = (float(r["gain"]), float(r["frac_short"]), float(r["r"]), float(r["u"]))
            elif r["part"] == "U":
                U[(ds, k, s, r["E"], int(r["K"]), r["group"])] = float(r["u"])
            elif r["part"] == "B1":
                B1[(ds, k, s, r["V"], r["F"], int(r["l"]), r["group"])] = (float(r["p"]), float(r["frac_single"]))
    return G, U, B1


def lstar(G, ds, k, E, V, F, K=10):
    best, bl = None, None
    for lv in range(1, 6):
        xs = [G.get((ds, k, s, E, V, F, lv, K, "valid")) for s in SEEDS]
        if any(x is None for x in xs):
            return None
        m = np.mean([x[0] for x in xs])
        if best is None or m > best:
            best, bl = m, lv
    return bl


def b_report():
    D = load_b()
    G, U, B1 = b_index(D)
    gates = json.load(open(os.path.join(OUT, "gates_b.json")))
    VS = ["X", "AX", "A2X", "rand1", "SELF", "SHUF"]
    have = lambda ds, k, E, V, F, K=10: all((ds, k, s, E, V, F, 1, K, "test") in G for s in SEEDS)

    def gain_at(ds, k, E, V, F, K=10, g="test"):
        lv = lstar(G, ds, k, E, V, F, 10)
        if lv is None:
            return None, None, None
        xs = [G[(ds, k, s, E, V, F, lv, K, g)] for s in SEEDS]
        return [x[0] for x in xs], lv, float(np.mean([x[1] for x in xs]))

    L = ["# T23b 요약 — 클러스터 support 순도와 제한 kNN (학습 없음)", ""]
    L.append("- gain = 그룹 평균 r_K − 그룹 평균 u_K(같은 anchor). ℓ* = valid gain(K = 10) seed 평균 최대 수준(동률이면 작은 ℓ). 값 = seed 평균 ± SE (부호 일치/3). "
             "SELF = 임베딩 자신의 배정, SHUF = AX 배정의 노드 무작위 재배정(17300 + s).")
    L.append(f"- 완료된 run 파일: {len(D)}/18.")
    L.append("")
    # B1
    L.append("## B1 — B-1 클러스터 support 순도 p (seed 평균 ± sd)")
    for ds, k in CELLS:
        if not all((ds, k, s) in D for s in SEEDS):
            L.append(f"- {ds} {k}-shot: 미실행")
            continue
        L.append(f"### {ds} {k}-shot")
        L.append("| 뷰 | 계열 | " + " | ".join(f"{g} ℓ{lv}" for g in GROUPS for lv in range(1, 6)) + " |")
        L.append("|---|---|" + "---|" * 15)
        for V in FIXED + ["rand1", "TEGe", "CONe"]:
            for F in ("H", "M"):
                cols = []
                for g in GROUPS:
                    for lv in range(1, 6):
                        xs = [B1[(ds, k, s, V, F, lv, g)][0] for s in SEEDS]
                        cols.append(f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1):.3f}")
                L.append(f"| {V} | {F} | " + " | ".join(cols) + " |")
        L.append("")
    L.append("- 클러스터 크기 중앙값(H 계열, seed 0 배정; ℓ1–ℓ5):")
    for ds in DATASETS:
        for V, k in [(v, None) for v in FIXED + ["rand1"]] + [(E, kk) for E in ("TEGe", "CONe") for kk in SHOTS]:
            H = load_assign(V, ds, k, 0, "H")
            med = [float(np.median(np.unique(H[:, lv], return_counts=True)[1])) for lv in range(5)]
            L.append(f"  - {ds} {V}{'' if k is None else f' {k}-shot'}: " + ", ".join(f"{m:g}" for m in med))
    L.append("")
    # B2
    L.append("## B2 — 기준 kNN 순도 u_K (seed 평균 ± sd)")
    L.append("| 데이터셋 | E | 셀 | 그룹 | K=5 | K=10 | K=20 | K=50 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        t19 = json.load(open(os.path.join(ROOT, "results", "T19a", f"{ds}.json")))["fixed"]
        for V in ("X", "AX"):
            for g in GROUPS:
                cols = [np.array([t19[f"{V}|{s}"][g][str(K)]["pur_all"] for s in SEEDS]) for K in (5, 10, 20, 50)]
                L.append(f"| {ds} | {V} (T19a) | — | {g} | " + " | ".join(f"{c.mean():.4f} ± {c.std(ddof=1):.4f}" for c in cols) + " |")
        for k in SHOTS:
            for E in ("TEGe", "CONe"):
                for g in GROUPS:
                    if not all((ds, k, s, E, 10, g) in U for s in SEEDS):
                        continue
                    cols = [np.array([U[(ds, k, s, E, K, g)] for s in SEEDS]) for K in (5, 10, 20, 50)]
                    L.append(f"| {ds} | {E} | {k} | {g} | " + " | ".join(f"{c.mean():.4f} ± {c.std(ddof=1):.4f}" for c in cols) + " |")
    L.append("")

    def gain_table(F, K, title):
        L.append(title)
        L.append("| E | V | " + " | ".join(f"{ds} {k}" for ds, k in CELLS) + " |")
        L.append("|---|---|" + "---|" * 6)
        for E in ("TEGe", "CONe"):
            for V in VS:
                cols = []
                for ds, k in CELLS:
                    if not have(ds, k, E, V, F, K):
                        cols.append("미실행")
                        continue
                    xs, lv, fs = gain_at(ds, k, E, V, F, K)
                    cols.append(f"{fmt_ms(xs)}, ℓ*={lv}, short={fs:.3f}")
                L.append(f"| {E} | {V} | " + " | ".join(cols) + " |")
        L.append("")

    gain_table("H", 10, "## B3 — 주 표: test gain, K = 10, F = H (ℓ*)")
    gain_table("M", 10, "## B4 — test gain, K = 10, F = M (ℓ*)")
    L.append("## B5 — 수준별 gain (E = TEGe, F = H, K = 10; seed 평균 ± SE)")
    L.append("| 셀 | V | 그룹 | ℓ1 | ℓ2 | ℓ3 | ℓ4 | ℓ5 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for V in VS:
            if not have(ds, k, "TEGe", V, "H"):
                continue
            for g in ("valid", "test"):
                L.append(f"| {ds} {k} | {V} | {g} | " + " | ".join(fmt_ms([G[(ds, k, s, 'TEGe', V, 'H', lv, 10, g)][0] for s in SEEDS]) for lv in range(1, 6)) + " |")
    L.append("")
    gain_table("H", 5, "## B6 — test gain, K = 5, F = H (ℓ*는 K = 10 기준, 기록)")

    R = {}
    cnt = lambda fn: sum(1 for ds, k in CELLS if fn(ds, k))
    g10 = lambda ds, k, E, V, F="H": gain_at(ds, k, E, V, F)[0] if have(ds, k, E, V, F) else None
    R["R1"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "AX") is not None and clear_pos(g10(ds, k, "TEGe", "AX")))
    R["R2"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "X") is not None and clear_pos(g10(ds, k, "TEGe", "X")))
    R["R3"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "SELF") is not None and clear_pos(g10(ds, k, "TEGe", "SELF")))
    R["R4"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "SHUF") is not None and clear_neg(g10(ds, k, "TEGe", "SHUF")))
    R["R5"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "AX") is not None and g10(ds, k, "CONe", "AX") is not None and
                  clear_pos(np.array(g10(ds, k, "TEGe", "AX")) - np.array(g10(ds, k, "CONe", "AX"))))
    R["R6"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "rand1") is not None and clear_pos(g10(ds, k, "TEGe", "rand1")))
    R["R7"] = cnt(lambda ds, k: g10(ds, k, "TEGe", "AX", "M") is not None and
                  clear_pos(np.array(g10(ds, k, "TEGe", "AX", "H")) - np.array(g10(ds, k, "TEGe", "AX", "M"))))
    dist = {}
    for ds, k in CELLS:
        for E in ("TEGe", "CONe"):
            for V in VS:
                for F in ("H", "M"):
                    lv = lstar(G, ds, k, E, V, F) if have(ds, k, E, V, F) else None
                    if lv:
                        dist[lv] = dist.get(lv, 0) + 1
    R["R7_lstar_dist"] = dict(sorted(dist.items()))
    L.append("## 판정 (값만; test, K = 10, F = H, ℓ*; 분명 = seed 평균 > 0, > 2·SE, 부호 3/3)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    for key, txt in (("R1", "gain(TEGe, AX) 분명히 양수"), ("R2", "gain(TEGe, X) 분명히 양수"), ("R3", "gain(TEGe, SELF) 분명히 양수"),
                     ("R4", "gain(TEGe, SHUF) 분명히 음수"), ("R5", "gain(TEGe, AX) − gain(CONe, AX) 분명히 양수"),
                     ("R6", "gain(TEGe, rand1) 분명히 양수"), ("R7", "(기록) gain_H(TEGe, AX) − gain_M(TEGe, AX) 분명히 양수")):
        L.append(f"| {key} | {txt} | {R[key]}/6 |")
    L.append(f"| R7 ℓ* 분포 (기록) | (E, V, F, 셀) 전체 | {R['R7_lstar_dist']} |")
    L.append("")
    P = [("P116", "R1 ≥ 4/6", R["R1"] >= 4, f"R1 = {R['R1']}/6"), ("P117", "R3 ≤ 1/6", R["R3"] <= 1, f"R3 = {R['R3']}/6"),
         ("P118", "R4 = 6/6", R["R4"] == 6, f"R4 = {R['R4']}/6"), ("P119", "R5 ≥ 4/6", R["R5"] >= 4, f"R5 = {R['R5']}/6"),
         ("P120", "R6 ≤ 2/6", R["R6"] <= 2, f"R6 = {R['R6']}/6")]
    L.append("## 사전 등록 대조")
    L.append("| 예측 | 조건 | 관측 | 일치 |")
    L.append("|---|---|---|---|")
    for p, c, ok, obs in P:
        L.append(f"| {p} | {c} | {obs} | {'일치' if ok else '어긋남'} |")
    L.append("")
    vals = [x for v in G.values() for x in v] + list(U.values()) + [v[0] for v in B1.values()]
    fin = bool(np.all(np.isfinite(vals)))
    L.append(f"- He: 모든 값 유한 = {fin} (값 {len(vals)}개).")
    open(os.path.join(ROOT, "reports", "T23b_summary.md"), "w").write("\n".join(L) + "\n")
    json.dump({"R": R, "P": {p: ok for p, _, ok, _ in P}, "He_finite": fin, "n_runs": len(D), "gates": gates},
              open(os.path.join(OUT, "judgements_b.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(R, ensure_ascii=False), {p: ok for p, _, ok, _ in P}, fin)


# ------------------------------------------------------------------ T23c
BETAS = [0.0, 0.05, 0.1, 0.2, 0.5]


def agree_counts(A, u, v, F_rows=None):
    """c = number of levels (1-5) where u and v share a cluster; s = 1[same cluster at level 5]."""
    eq = A[u] == A[v]
    return eq.sum(1), eq[:, 4].astype(np.int64)


def fit_auc(Xtr, ytr, Xte, yte):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit((Xtr - mu) / sd, ytr)
        conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd))), conv


def make_episodes(rng, classes, by_class, n=200):
    cands = [c for c in classes if len(by_class[c]) >= 5]
    eps = []
    for _ in range(n):
        cl = rng.choice(len(cands), 5, replace=False)
        eps.append(np.concatenate([rng.choice(by_class[cands[i]], 5, replace=False) for i in cl]))
    return np.stack(eps), len(cands)


def episode_auc(S, cls):
    """S [B, 25, 25] scores, cls [25] class index per position. Mean over episodes and supports of the AUC that ranks the
    4 same-class others above the 20 others (ties 0.5, Mann-Whitney)."""
    from scipy.stats import rankdata

    B = S.shape[0]
    off = ~np.eye(25, dtype=bool)
    Ss = S[:, off].reshape(B, 25, 24)
    pos = (cls[:, None] == cls[None, :])[off].reshape(25, 24)
    rk = rankdata(Ss, axis=2)
    rp = (rk * pos[None]).sum(2)
    auc = (rp - 4 * 5 / 2) / (4 * 20)
    return float(auc.mean())


def c_run(ds, k):
    import csv

    import analyze_t19 as a19
    from sklearn.metrics import roc_auc_score

    t_all = time.time()
    od = os.path.join(OUT, "T23c", f"{ds}_5w{k}s")
    os.makedirs(od, exist_ok=True)
    timing, gate_id, gate_ib = {}, [], []
    for s in SEEDS:
        t0 = time.time()
        sp, lab = a19.split_of(emb_dir("TEGe", ds, k, s))
        gid = a19.group_ids(sp, lab)
        y = labels_eff(lab, gid)
        anchors = np.where(gid >= 0)[0]
        gof = gid[anchors]
        A = {F: {v: load_assign(v, ds, k, s, F) for v in FIXED + ["rand1", "TEGe", "CONe"]} for F in ("H", "M")}
        for F in ("H", "M"):
            A[F]["SHUF"] = shuf_of(A[F]["AX"], s)[0]
        VS = ["X", "AX", "A2X", "rand1", "SELF", "SHUF"]
        rows = []
        # episodes (shared by E, V, F)
        by_class = {}
        for c in np.unique(lab[anchors]):
            by_class[int(c)] = np.where(lab == c)[0]
        eps = {}
        for split, off, key in (("test", 17100, "class_list_test"), ("valid", 17200, "class_list_valid")):
            classes = [int(c) for c in sp[key]]
            E_, ncand = make_episodes(np.random.default_rng(off + s), classes, by_class)
            eps[split] = E_
            ok = all(set(int(lab[n]) for n in e) <= set(classes) and len(set(e.tolist())) == 25 and
                     all(int((lab[e] == c).sum()) in (0, 5) for c in set(lab[e].tolist())) and len(set(lab[e].tolist())) == 5 for e in E_)
            gate_id.append({"seed": s, "split": split, "ok": bool(ok), "n_class_cand": ncand})
        if s == 0:
            np.savez(os.path.join(od, "episodes_seed0.npz"), test=eps["test"], valid=eps["valid"])
        cls = np.repeat(np.arange(5), 5)
        for E in ("TEGe", "CONe"):
            Z = np.load(os.path.join(emb_dir(E, ds, k, s), "emb_best.npy"))
            Zn = a19.normed(Z)
            nb, n = restricted_knn(Zn, anchors, None, K=50)
            U = np.repeat(anchors, 50)
            Vv = nb.reshape(-1)
            cos = (Zn[torch.as_tensor(U, device=DEV)] * Zn[torch.as_tensor(Vv, device=DEV)]).sum(1).cpu().numpy()
            ypair = (y[U] == y[Vv]).astype(np.int64)
            gpair = np.repeat(gof, 50)
            # Ib: K = 10 truncation, test mean
            t10 = (y[nb[:, :10]] == y[anchors][:, None]).mean(1)
            gate_ib.append({"seed": s, "E": E, "u10_test": float(t10[gof == 2].mean())})
            sel = {g: gpair == gi for gi, g in enumerate(GROUPS)}
            auc_cos = {g: float(roc_auc_score(ypair[sel[g]], cos[sel[g]])) for g in ("valid", "test")}
            Zne = torch.nn.functional.normalize(torch.as_tensor(Z, dtype=torch.float64), dim=1).numpy()
            for F in ("H", "M"):
                for V in VS:
                    AA = A[F][E if V == "SELF" else V]
                    c, sg = agree_counts(AA, U, Vv)
                    res = {}
                    for name, f in (("c", c), ("s", sg)):
                        Xf = np.stack([cos, f.astype(np.float64)], 1)
                        for g in ("valid", "test"):
                            res[(name, g)] = fit_auc(Xf[sel["base"]], ypair[sel["base"]], Xf[sel[g]], ypair[sel[g]])
                    for g in ("valid", "test"):
                        rows.append({"part": "C1", "E": E, "V": V, "F": F, "group": g, "auc_cos": auc_cos[g], "auc_c": res[("c", g)][0],
                                     "auc_s": res[("s", g)][0], "d_multi": res[("c", g)][0] - auc_cos[g], "d_single": res[("s", g)][0] - auc_cos[g],
                                     "conv_c": res[("c", g)][1], "conv_s": res[("s", g)][1]})
                    if E == "TEGe" and V == "AX" and F == "H" and s == 0:
                        te = sel["test"]
                        q = np.quantile(cos[te], np.linspace(0, 1, 11))
                        dec = np.clip(np.searchsorted(q, cos[te], side="right") - 1, 0, 9)
                        for di in range(10):
                            for cv in range(6):
                                m = (dec == di) & (c[te] == cv)
                                rows.append({"part": "C3", "E": E, "V": V, "F": F, "decile": di + 1, "c": cv, "n": int(m.sum()),
                                             "frac_same": float(ypair[te][m].mean()) if m.sum() else float("nan")})
                    # C-2
                    for split in ("valid", "test"):
                        ep = eps[split]
                        Ze = Zne[ep]
                        COS = np.einsum("bid,bjd->bij", Ze, Ze)
                        Ae = AA[ep]
                        Cm = (Ae[:, :, None, :] == Ae[:, None, :, :]).sum(3).astype(np.float64)
                        rows.append({"part": "C2", "E": E, "V": V, "F": F, "split": split, "score": "S1", "auc": episode_auc(Cm, cls)})
                        for b in BETAS:
                            rows.append({"part": "C2", "E": E, "V": V, "F": F, "split": split, "score": f"S2_{b:g}",
                                         "auc": episode_auc(COS + b * Cm / 5.0, cls)})
            del Zn
            torch.cuda.empty_cache()
        keys = sorted({kk for r in rows for kk in r})
        with open(os.path.join(od, f"seed{s}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        timing[s] = time.time() - t0
        print(ds, k, s, f"{timing[s]:.1f}s", flush=True)
    json.dump({"sec": time.time() - t_all, "seed_sec": timing, "Id": gate_id, "Ib": gate_ib}, open(os.path.join(od, "meta.json"), "w"), indent=1)


def c_gates():
    out = {}
    # Ic (a): synthetic non-random assignment, 50 nodes
    n = 50
    A = np.stack([np.arange(n) // 25, np.arange(n) // 10, np.arange(n) % 7, np.arange(n) // 3, np.arange(n) % 11], 1)
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    u, v = u.ravel(), v.ravel()
    c, sg = agree_counts(A, u, v)
    cb = np.array([sum(int(A[a, l] == A[b, l]) for l in range(5)) for a, b in zip(u, v)])
    sb = np.array([int(A[a, 4] == A[b, 4]) for a, b in zip(u, v)])
    out["Ic_synth"] = {"c_mismatch": int((c != cb).sum()), "s_mismatch": int((sg != sb).sum()), "pairs": int(len(u))}
    # Ic (b): real H assignment, c = deepest shared level
    H = load_assign("AX", "Amazon_clothing", None, 0, "H")
    ar = np.arange(200000, dtype=np.int64)  # deterministic pair list (no RNG)
    uu, vv = (ar * 7919) % len(H), (ar * 104729 + 13) % len(H)
    c2, _ = agree_counts(H, uu, vv)
    eq = H[uu] == H[vv]
    deepest = np.where(eq.any(1), 5 - np.argmax(eq[:, ::-1], axis=1), 0)
    out["Ic_real"] = {"mismatch": int((c2 != deepest).sum()), "pairs": 200000}
    out["pass"] = out["Ic_synth"]["c_mismatch"] == 0 and out["Ic_synth"]["s_mismatch"] == 0 and out["Ic_real"]["mismatch"] == 0
    json.dump(out, open(os.path.join(OUT, "gates_c.json"), "w"), indent=1)
    print(json.dumps(out))


def c_report():
    import csv

    VS = ["X", "AX", "A2X", "rand1", "SELF", "SHUF"]
    D, metas = {}, {}
    for ds, k in CELLS:
        mp = os.path.join(OUT, "T23c", f"{ds}_5w{k}s", "meta.json")
        if os.path.exists(mp):
            metas[(ds, k)] = json.load(open(mp))
        for s in SEEDS:
            p = os.path.join(OUT, "T23c", f"{ds}_5w{k}s", f"seed{s}.csv")
            if os.path.exists(p):
                D[(ds, k, s)] = list(csv.DictReader(open(p)))
    C1, C2, C3 = {}, {}, {}
    for (ds, k, s), rows in D.items():
        for r in rows:
            if r["part"] == "C1":
                C1[(ds, k, s, r["E"], r["V"], r["F"], r["group"])] = {x: float(r[x]) for x in ("auc_cos", "auc_c", "auc_s", "d_multi", "d_single")} | \
                    {"conv": r["conv_c"] == "True" and r["conv_s"] == "True"}
            elif r["part"] == "C2":
                C2[(ds, k, s, r["E"], r["V"], r["F"], r["split"], r["score"])] = float(r["auc"])
            elif r["part"] == "C3":
                C3[(ds, k, int(r["decile"]), int(r["c"]))] = (float(r["frac_same"]), int(r["n"]))
    full = lambda ds, k: all((ds, k, s) in D for s in SEEDS)
    sv = lambda ds, k, E, V, F, key, g="test": [C1[(ds, k, s, E, V, F, g)][key] for s in SEEDS]

    def bstar(ds, k, E, V, F):
        best, bb = None, None
        for b in BETAS:
            m = np.mean([C2[(ds, k, s, E, V, F, "valid", f"S2_{b:g}")] for s in SEEDS])
            if best is None or m > best:
                best, bb = m, b
        return bb

    def dep(ds, k, E, V, F):
        b = bstar(ds, k, E, V, F)
        return [C2[(ds, k, s, E, V, F, "test", f"S2_{b:g}")] - C2[(ds, k, s, E, V, F, "test", "S2_0")] for s in SEEDS], b

    L = ["# T23c 요약 — 일치 수 c의 추가 신호 (학습 없음)", ""]
    L.append("- C-1: 그룹 anchor의 E top-50 이웃 쌍. AUC_cos = test 쌍 코사인 AUC. AUC_cos+c / cos+s = base 쌍으로 맞춘 로지스틱 회귀(표준화, lbfgs, C = 1)를 test 쌍에 적용. "
             "ΔAUC_multi = AUC_cos+c − AUC_cos, ΔAUC_single = AUC_cos+s − AUC_cos. 값 = seed 평균 ± SE (부호 일치/3).")
    L.append("- C-2: test(valid) 클래스 5-way 5-shot 에피소드 200개(`default_rng(17100 + s)` / `17200 + s`), support 25개 중 하나가 같은 클래스 4개를 나머지 20개보다 높게 매기는 AUC. "
             "S0 = cos, S1 = c, S2(β) = cos + β·c/5. β* = valid S2 seed 평균 최대(동률이면 작은 β).")
    L.append(f"- 완료된 run 파일: {len(D)}/18.")
    L.append("")

    def c1_table(F, title):
        L.append(title)
        L.append("| E | V | " + " | ".join(f"{ds} {k}" for ds, k in CELLS) + " |")
        L.append("|---|---|" + "---|" * 6)
        for E in ("TEGe", "CONe"):
            for V in VS:
                cols = []
                for ds, k in CELLS:
                    if not full(ds, k) or (ds, k, 0, E, V, F, "test") not in C1:
                        cols.append("미실행")
                        continue
                    cols.append(f"AUC_cos {np.mean(sv(ds, k, E, V, F, 'auc_cos')):.4f}; Δmulti {fmt_ms(sv(ds, k, E, V, F, 'd_multi'))}; "
                                f"Δsingle {fmt_ms(sv(ds, k, E, V, F, 'd_single'))}")
                L.append(f"| {E} | {V} | " + " | ".join(cols) + " |")
        L.append("")

    c1_table("H", "## C1 — 주 표: C-1 노드 쌍, test, F = H")
    c1_table("M", "## C2 — C-1 노드 쌍, test, F = M")
    L.append("## C3 — 코사인 10분위 × c → 같은-클래스 비율 (쌍 수) (E = TEGe, V = AX, F = H, seed 0, test 쌍)")
    for ds, k in CELLS:
        if (ds, k, 1, 0) not in C3:
            L.append(f"- {ds} {k}-shot: 미실행")
            continue
        L.append(f"### {ds} {k}-shot")
        L.append("| 분위 | " + " | ".join(f"c={c}" for c in range(6)) + " |")
        L.append("|---|" + "---|" * 6)
        for d in range(1, 11):
            L.append(f"| {d} | " + " | ".join(("—" if C3[(ds, k, d, c)][1] == 0 else f"{C3[(ds, k, d, c)][0]:.3f} ({C3[(ds, k, d, c)][1]})") for c in range(6)) + " |")
        L.append("")
    L.append("## C4 — C-2 에피소드, test, F = H: AUC(S0) / AUC(S1) / AUC(S2(β*)) / β* / ΔAUC_ep")
    L.append("| E | V | " + " | ".join(f"{ds} {k}" for ds, k in CELLS) + " |")
    L.append("|---|---|" + "---|" * 6)
    for E in ("TEGe", "CONe"):
        for V in VS:
            cols = []
            for ds, k in CELLS:
                if not full(ds, k) or (ds, k, 0, E, V, "H", "test", "S1") not in C2:
                    cols.append("미실행")
                    continue
                d, b = dep(ds, k, E, V, "H")
                m = lambda sc: np.mean([C2[(ds, k, s, E, V, "H", "test", sc)] for s in SEEDS])
                cols.append(f"{m('S2_0'):.4f} / {m('S1'):.4f} / {m(f'S2_{b:g}'):.4f} / β*={b:g} / {fmt_ms(d)}")
            L.append(f"| {E} | {V} | " + " | ".join(cols) + " |")
    L.append("")
    L.append("## C5 — β 곡선 (E = TEGe, V = AX, F = H; seed 평균)")
    L.append("| 셀 | 분할 | " + " | ".join(f"β={b:g}" for b in BETAS) + " |")
    L.append("|---|---|" + "---|" * len(BETAS))
    for ds, k in CELLS:
        if not full(ds, k):
            continue
        for split in ("valid", "test"):
            L.append(f"| {ds} {k} | {split} | " + " | ".join(f"{np.mean([C2[(ds, k, s, 'TEGe', 'AX', 'H', split, f'S2_{b:g}')] for s in SEEDS]):.4f}" for b in BETAS) + " |")
    L.append("")
    cnt = lambda fn: sum(1 for ds, k in CELLS if full(ds, k) and fn(ds, k))
    S = {}
    S["S1"] = cnt(lambda ds, k: clear_pos(sv(ds, k, "TEGe", "AX", "H", "d_multi")))
    S["S2"] = cnt(lambda ds, k: clear_pos(np.array(sv(ds, k, "TEGe", "AX", "H", "d_multi")) - np.array(sv(ds, k, "TEGe", "AX", "H", "d_single"))))
    S["S3"] = cnt(lambda ds, k: clear_pos(sv(ds, k, "TEGe", "SELF", "H", "d_multi")))
    S["S4"] = cnt(lambda ds, k: (ds, k, 0, "CONe", "AX", "H", "test") in C1 and
                  clear_pos(np.array(sv(ds, k, "TEGe", "AX", "H", "d_multi")) - np.array(sv(ds, k, "CONe", "AX", "H", "d_multi"))))
    S["S5"] = cnt(lambda ds, k: clear_pos(dep(ds, k, "TEGe", "AX", "H")[0]))
    S["S6_X"] = cnt(lambda ds, k: clear_pos(sv(ds, k, "TEGe", "X", "H", "d_multi")))
    S["S6_rand1"] = cnt(lambda ds, k: clear_pos(sv(ds, k, "TEGe", "rand1", "H", "d_multi")))
    S["S6_SHUF"] = cnt(lambda ds, k: clear_pos(sv(ds, k, "TEGe", "SHUF", "H", "d_multi")))
    bd = {}
    for ds, k in CELLS:
        if not full(ds, k):
            continue
        for E in ("TEGe", "CONe"):
            for V in VS:
                for F in ("H", "M"):
                    if (ds, k, 0, E, V, F, "valid", "S2_0") in C2:
                        b = bstar(ds, k, E, V, F)
                        bd[b] = bd.get(b, 0) + 1
    S["S6_beta_dist"] = {f"{b:g}": n for b, n in sorted(bd.items())}
    L.append("## 판정 (값만; test, F = H; 분명 = seed 평균 > 0, > 2·SE, 부호 3/3)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    for key, txt in (("S1", "ΔAUC_multi(TEGe, AX) 분명히 양수"), ("S2", "ΔAUC_multi(TEGe, AX) − ΔAUC_single(TEGe, AX) 분명히 양수"),
                     ("S3", "ΔAUC_multi(TEGe, SELF) 분명히 양수"), ("S4", "ΔAUC_multi(TEGe, AX) − ΔAUC_multi(CONe, AX) 분명히 양수"),
                     ("S5", "ΔAUC_ep(TEGe, AX) 분명히 양수")):
        L.append(f"| {key} | {txt} | {S[key]}/6 |")
    L.append(f"| S6 (기록) | ΔAUC_multi(TEGe, X / rand1 / SHUF) 분명히 양수 | {S['S6_X']}/6 / {S['S6_rand1']}/6 / {S['S6_SHUF']}/6 |")
    L.append(f"| S6 β* 분포 (기록) | (E, V, F, 셀) 전체 | {S['S6_beta_dist']} |")
    L.append("")
    P = [("P121", "S1 ≥ 4/6", S["S1"] >= 4, S["S1"]), ("P122", "S2 ≥ 4/6", S["S2"] >= 4, S["S2"]), ("P123", "S3 ≤ 1/6", S["S3"] <= 1, S["S3"]),
         ("P124", "S4 ≥ 4/6", S["S4"] >= 4, S["S4"]), ("P125", "S5 ≥ 4/6", S["S5"] >= 4, S["S5"])]
    L.append("## 사전 등록 대조")
    L.append("| 예측 | 조건 | 관측 | 일치 |")
    L.append("|---|---|---|---|")
    for p, c, ok, v in P:
        L.append(f"| {p} | {c} | {p.replace('P121', 'S1').replace('P122', 'S2').replace('P123', 'S3').replace('P124', 'S4').replace('P125', 'S5')} = {v}/6 | {'일치' if ok else '어긋남'} |")
    L.append("")
    # gates Ib, Id, Ie
    Ub = {}
    for ds, k in CELLS:
        for s in SEEDS:
            p = os.path.join(OUT, "T23b", f"{ds}_5w{k}s", f"seed{s}.csv")
            if os.path.exists(p):
                for r in csv.DictReader(open(p)):
                    if r["part"] == "U" and r["K"] == "10" and r["group"] == "test":
                        Ub[(ds, k, s, r["E"])] = float(r["u"])
    ib = [abs(x["u10_test"] - Ub[(ds, k, x["seed"], x["E"])]) for (ds, k), m in metas.items() for x in m["Ib"]]
    idok = all(x["ok"] for m in metas.values() for x in m["Id"])
    vals = [v for r in C1.values() for kk, v in r.items() if kk != "conv"] + list(C2.values())
    nconv = sum(1 for r in C1.values() if not r["conv"])
    G = {"Ib_max_absdiff": max(ib) if ib else None, "Ib_n": len(ib), "Id": idok, "Id_n_class_cand": sorted({(x["split"], x["n_class_cand"]) for m in metas.values() for x in m["Id"]}),
         "Ie_finite": bool(np.all(np.isfinite(vals))), "Ie_nonconverged": nconv, "n_runs": len(D)}
    L.append(f"- Ib: \|C-1 top-10 test 순도 − T23b u_10\| 최대 {G['Ib_max_absdiff']} ({G['Ib_n']}개). Id: {idok}. Ie: 유한 {G['Ie_finite']}, 회귀 미수렴 {nconv}개.")
    open(os.path.join(ROOT, "reports", "T23c_summary.md"), "w").write("\n".join(L) + "\n")
    G["Id_n_class_cand"] = [list(x) for x in G["Id_n_class_cand"]]
    json.dump({"S": S, "P": {p: ok for p, _, ok, _ in P}, "gates": G}, open(os.path.join(OUT, "judgements_c.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, ensure_ascii=False), {p: ok for p, _, ok, _ in P}, json.dumps(G))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "a-run":
        a_run(sys.argv[2], int(sys.argv[3]))
    elif cmd == "a-gates":
        a_gates()
    elif cmd == "a-report":
        a_report()
    elif cmd == "b-run":
        b_run(sys.argv[2], int(sys.argv[3]))
    elif cmd == "b-gates":
        b_gates()
    elif cmd == "b-report":
        b_report()
    elif cmd == "c-run":
        c_run(sys.argv[2], int(sys.argv[3]))
    elif cmd == "c-gates":
        c_gates()
    elif cmd == "c-report":
        c_report()
