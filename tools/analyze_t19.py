"""T19a (E1 pseudo-class purity) / T19b (E2 sensor check). Analysis only, no training.

Views: X (utils.load_data), A_hat X = fusion_baseline.diffusion_features_hops(ds, 1), A_hat^2 X = fusion_baseline.diffusion_features(ds),
rand1 = analyze_t06b.rand_spaces(analyze_t06b.dataset_spaces(ds), seed)["rand1"] (offset 2000), 원본 = T14 base emb_best.npy,
대조 = T14 uni emb_best.npy. Cosine after row L2 normalization; self excluded.
kNN tie rule (user decision 2026-10-11): similarities in float64 on the GPU, top-(K_max + 1 + margin) candidates moved to the
CPU and ordered by (similarity desc, node index asc). Neighbour indices stay in memory (not saved).

Usage:
  python tools/analyze_t19.py t19a-gates          # Gb, Gc (Ga is checked per run in t19a-run)
  python tools/analyze_t19.py t19a-run <dataset>  # one dataset, all views / runs -> results/T19a/<dataset>.json
  python tools/analyze_t19.py t19a-report         # tables, judgements -> reports/T19a_summary.md, results/T19a/gate_checks.json
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
R14 = os.path.join(ROOT, "results", "T14")
OUT19A = os.path.join(ROOT, "results", "T19a")
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(5))
GROUPS = ["base", "valid", "test"]
KS = [1, 5, 10, 20, 50]
FIXED = ["X", "AX", "A2X"]
RUNV = {"원본": "base", "대조": "uni"}
MARGIN = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ kNN with the tie rule
def knn_sorted(Zn, anchors, kmax, cand=None, margin=MARGIN, block=4096):
    """Zn: [N, d] float64 torch (row-normalized, on DEV). anchors: global ids (np). cand: global candidate ids (np) or None (all N).
    Returns (idx [n, kmax+1] global ids, sim [n, kmax+1], n_margin_short): the first kmax+1 neighbours ordered by
    (sim desc, index asc). n_margin_short = anchors whose last fetched candidate ties the (kmax+1)-th value; those rows are
    re-ordered over all candidates."""
    C = Zn if cand is None else Zn[torch.as_tensor(cand, device=Zn.device)]
    cand_np = np.arange(Zn.shape[0]) if cand is None else np.asarray(cand)
    pos = np.full(Zn.shape[0], -1, dtype=np.int64)
    pos[cand_np] = np.arange(len(cand_np))
    m = min(kmax + 1 + margin, len(cand_np) - 1)
    out_i, out_v, short = [], [], 0
    for s in range(0, len(anchors), block):
        a = np.asarray(anchors[s : s + block])
        sim = Zn[torch.as_tensor(a, device=Zn.device)] @ C.T
        pa = pos[a]
        has = np.where(pa >= 0)[0]
        if len(has):
            sim[torch.as_tensor(has, device=sim.device), torch.as_tensor(pa[has], device=sim.device)] = float("-inf")
        v, i = sim.topk(m, dim=1)
        v, i = v.cpu().numpy(), i.cpu().numpy()
        g = cand_np[i]
        order = np.lexsort((g, -v), axis=1)
        v = np.take_along_axis(v, order, 1)
        g = np.take_along_axis(g, order, 1)
        last = v[:, -1].copy()
        v, g = v[:, : kmax + 1].copy(), g[:, : kmax + 1].copy()
        if m > kmax + 1:
            # ties reaching past the fetched candidates: re-order those rows over all candidates (exact tie rule)
            rows = np.where(last == v[:, -1])[0]
            short += int(len(rows))
            for r in rows:
                full = sim[r].cpu().numpy()
                o = np.lexsort((cand_np, -full))[: kmax + 1]
                v[r], g[r] = full[o], cand_np[o]
        out_i.append(g)
        out_v.append(v)
    return np.concatenate(out_i), np.concatenate(out_v), short


def normed(Z):
    t = torch.as_tensor(np.asarray(Z), dtype=torch.float64, device=DEV)
    return torch.nn.functional.normalize(t, dim=1)


# ------------------------------------------------------------------ views
def fixed_views(ds):
    import fusion_baseline as fb
    from utils import load_data

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        X = load_data(ds)[2].numpy()
    finally:
        os.chdir(cwd)
    return {"X": X, "AX": fb.diffusion_features_hops(ds, 1), "A2X": fb.diffusion_features(ds)}


def rand1_views(ds):
    from analyze_t06b import dataset_spaces, rand_spaces  # import side effect: os.chdir(ROOT)

    t0 = time.time()
    base = dataset_spaces(ds)
    t_ds = time.time() - t0
    return {s: rand_spaces(base, s)["rand1"] for s in SEEDS}, t_ds


def run_dir(cfg, ds, k, s):
    return os.path.join(R14, cfg, f"{ds}_5w{k}s", f"seed{s}")


def split_of(d):
    sp = json.load(open(os.path.join(d, "split.json")))
    lab = np.load(os.path.join(d, "labels.npy"))
    return sp, lab


def group_ids(sp, lab):
    gid = np.full(len(lab), -1, dtype=np.int64)  # -1 = unassigned
    for gi, key in enumerate(("class_list_train", "class_list_valid", "class_list_test")):
        gid[np.isin(lab, [int(c) for c in sp[key]])] = gi
    return gid


def gate_split(ds, s):
    """Ga: labels.npy and split.json of every T14 base/uni run vs utils.load_data after seed_everything(s)."""
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
    for cfg in RUNV.values():
        for k in SHOTS:
            sp, lab = split_of(run_dir(cfg, ds, k, s))
            ok_lab = bool(np.array_equal(lab, labels))
            ok_list = [int(x) for x in sp["class_list_train"]] == [int(x) for x in tr] and \
                [int(x) for x in sp["class_list_valid"]] == [int(x) for x in va] and [int(x) for x in sp["class_list_test"]] == [int(x) for x in te]
            res.append({"cfg": cfg, "shot": k, "seed": s, "labels_equal": ok_lab, "lists_equal": ok_list})
    return res


# ------------------------------------------------------------------ T19a measures
def purity_stats(nb, sims, lab, gid, anchors_by_group, ks):
    """nb/sims rows aligned with all nodes (anchor i = node i). Returns {group: {K: {...}}}."""
    out = {}
    for gname, gi in zip(GROUPS, range(3)):
        a = anchors_by_group[gname]
        ya, ga = lab[a][:, None], gi
        n_lab = lab[nb[a]]
        n_grp = gid[nb[a]]
        d = {}
        for K in ks:
            d[K] = {"pur_all": float((n_lab[:, :K] == ya).mean()),
                    "frac_other": float((n_grp[:, :K] != ga).mean()),
                    "frac_unassigned": float((n_grp[:, :K] == -1).mean()),
                    "frac_tie": float((sims[a, K - 1] == sims[a, K]).mean())}
        out[gname] = d
    return out


def grp_purity(Zn, lab, members, K=10):
    nb, sims, short = knn_sorted(Zn, members, K, cand=members)
    return float((lab[nb[:, :K]] == lab[members][:, None]).mean()), short


def measure_view(Z, lab, gid, kmax=50):
    Zn = normed(Z)
    N = Zn.shape[0]
    nb, sims, short = knn_sorted(Zn, np.arange(N), kmax)
    abg = {g: np.where(gid == i)[0] for i, g in enumerate(GROUPS)}
    res = purity_stats(nb, sims, lab, gid, abg, KS)
    pg = {}
    for g in GROUPS:
        pg[g], sh = grp_purity(Zn, lab, abg[g])
        short += sh
    res["pur_grp10"] = pg
    res["margin_short"] = short
    del Zn
    torch.cuda.empty_cache()
    return res


def t19a_run(ds):
    os.makedirs(OUT19A, exist_ok=True)
    t0 = time.time()
    rec = {"dataset": ds, "ga": [], "fixed": {}, "rand1": {}, "runs": {}, "time": {}}
    for s in SEEDS:
        rec["ga"] += gate_split(ds, s)
    sp0, lab0 = split_of(run_dir("base", ds, 1, 0))
    fv = fixed_views(ds)
    rv, t_ds = rand1_views(ds)
    rec["time"]["dataset_spaces_sec"] = t_ds
    # fixed views do not depend on shot/seed, but the group split does (seed); measured per seed split
    for s in SEEDS:
        sp, lab = split_of(run_dir("base", ds, 1, s))
        gid = group_ids(sp, lab)
        for name, Z in fv.items():
            rec["fixed"][f"{name}|{s}"] = measure_view(Z, lab, gid)
        rec["rand1"][str(s)] = measure_view(rv[s], lab, gid)
        for k in SHOTS:
            for vname, cfg in RUNV.items():
                d = run_dir(cfg, ds, k, s)
                sp_k, lab_k = split_of(d)
                Z = np.load(os.path.join(d, "emb_best.npy"))
                rec["runs"][f"{vname}|{k}|{s}"] = {"shape": list(Z.shape)} | measure_view(Z, lab_k, group_ids(sp_k, lab_k))
        print(ds, "seed", s, "done", round(time.time() - t0), flush=True)
    rec["time"]["total_sec"] = time.time() - t0
    json.dump(rec, open(os.path.join(OUT19A, f"{ds}.json"), "w"))


# ------------------------------------------------------------------ gates Gb, Gc
def synthetic(n=200, d=16):
    """Deterministic (no RNG) matrix with duplicated rows: rows 0-19 are copied to 100-119 and 150-159."""
    i = np.arange(n)[:, None].astype(np.float64)
    j = np.arange(d)[None, :].astype(np.float64)
    Z = np.sin(1.3 * i + 0.7 * j + 0.01 * i * j) + 0.1 * np.cos(0.37 * i * (j + 1))
    Z[100:120] = Z[0:20]
    Z[150:160] = Z[0:10]
    lab = (np.arange(n) * 7) % 5
    gid = np.arange(n) % 3
    return Z, lab, gid


def brute_pur(Z, lab, anchors, cand, K):
    V = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    out = []
    for a in anchors:
        c = np.array([x for x in cand if x != a])
        s = V[c] @ V[a]
        o = np.lexsort((c, -s))
        out.append((lab[c[o[:K]]] == lab[a]).mean())
    return np.array(out)


def t19a_gates():
    from analyze_t13 import groups_of
    from analyze_t14 import knn_purity as kp14

    G = {}
    # Gb
    miss, shapes = [], {}
    for cfg in RUNV.values():
        for ds in DATASETS:
            for k in SHOTS:
                for s in SEEDS:
                    f = os.path.join(run_dir(cfg, ds, k, s), "emb_best.npy")
                    if not os.path.exists(f):
                        miss.append(f)
                        continue
                    shapes.setdefault((ds, np.load(f, mmap_mode="r").shape), 0)
                    shapes[(ds, np.load(f, mmap_mode="r").shape)] += 1
    G["Gb"] = {"missing": miss, "shapes": {f"{a}|{b}": v for (a, b), v in shapes.items()}, "n_found": 60 - len(miss)}
    # Gc (a) pur_grp K=10 vs analyze_t14.knn_purity
    d = run_dir("uni", "Amazon_clothing", 1, 0)
    Z = np.load(os.path.join(d, "emb_best.npy"))
    grp, sp, lab = groups_of(d)
    nodes, y = grp["test"]
    ref = kp14(Z, nodes, y)
    Zn = normed(Z)
    nb, sims, short = knn_sorted(Zn, nodes, 10, cand=nodes)
    mine_per = (lab[nb[:, :10]] == lab[nodes][:, None]).mean(1)
    mine = float(mine_per.mean())
    # per-anchor reference with the same selection code as analyze_t14.knn_purity (tools/analyze_t14.py:56-66)
    V = torch.nn.functional.normalize(torch.from_numpy(Z[nodes]).double(), dim=1)
    ref_per = []
    for s in range(0, len(nodes), 4096):
        sim = V[s : s + 4096] @ V.T
        r = torch.arange(sim.shape[0])
        sim[r, r + s] = float("-inf")
        ii = sim.topk(10, dim=1).indices
        ref_per.append((torch.from_numpy(y)[ii] == torch.from_numpy(y)[s : s + 4096, None]).double().mean(1).numpy())
    ref_per = np.concatenate(ref_per)
    diff = np.where(np.abs(ref_per - mine_per) > 0)[0]
    tie_at_boundary = sims[diff, 9] == sims[diff, 10]
    G["Gc_a"] = {"t14": ref, "t19": mine, "absdiff": abs(ref - mine), "n_anchor_mismatch": int(len(diff)),
                 "n_mismatch_boundary_tie": int(tie_at_boundary.sum()), "margin_short": short,
                 "pass": abs(ref - mine) <= 1e-6 or bool(tie_at_boundary.all()),
                 "basis": "pass_by_absdiff" if abs(ref - mine) <= 1e-6 else ("tie" if tie_at_boundary.all() else "fail")}
    # Gc (b) pur_all and pur_grp on the synthetic matrix vs brute force (same tie rule)
    Zs, labs, gids = synthetic()
    Zsn = normed(Zs)
    nb, sims, _ = knn_sorted(Zsn, np.arange(200), 50)
    ok_all = all(np.array_equal((labs[nb[:, :K]] == labs[:, None]).mean(1), brute_pur(Zs, labs, np.arange(200), np.arange(200), K)) for K in KS)
    gm = np.where(gids == 1)[0]
    nbg, _, _ = knn_sorted(Zsn, gm, 10, cand=gm)
    ok_grp = np.array_equal((labs[nbg[:, :10]] == labs[gm][:, None]).mean(1), brute_pur(Zs, labs, gm, gm, 10))
    n_ties = int((sims[:, :50] == sims[:, 1:51]).sum())
    # same check with margin 1 so that the full re-ordering path for ties past the fetched candidates is exercised
    nb1, _, sh1 = knn_sorted(Zsn, np.arange(200), 50, margin=1)
    ok_m1 = all(np.array_equal((labs[nb1[:, :K]] == labs[:, None]).mean(1), brute_pur(Zs, labs, np.arange(200), np.arange(200), K)) for K in KS)
    G["Gc_b"] = {"pur_all_equal_all_K": bool(ok_all), "pur_grp_equal": bool(ok_grp), "n_adjacent_equal_sims_in_top51": n_ties,
                 "margin1_pur_all_equal_all_K": bool(ok_m1), "margin1_rows_reordered": sh1, "pass": bool(ok_all and ok_grp and ok_m1)}
    os.makedirs(OUT19A, exist_ok=True)
    json.dump(G, open(os.path.join(OUT19A, "gates_bc.json"), "w"), indent=1)
    print(json.dumps(G, indent=1))


# ------------------------------------------------------------------ report
def ms(xs, fmt="{:.4f}"):
    xs = [x for x in xs if x is not None]
    m = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.format(sd)}", m


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    agree = int(np.sum(np.sign(d) == np.sign(m))) if m != 0 else 0
    return f"{m:+.4f} ± {se:.4f} ({agree}/{len(d)})", m, se


def t19a_report():
    R = {ds: json.load(open(os.path.join(OUT19A, f"{ds}.json"))) for ds in DATASETS}
    GBC = json.load(open(os.path.join(OUT19A, "gates_bc.json")))

    def vals(ds, k, view, g, K, metric="pur_all"):
        r = R[ds]
        if view in FIXED:
            return [r["fixed"][f"{view}|{s}"][g][str(K)][metric] for s in SEEDS]
        if view == "rand1":
            return [r["rand1"][str(s)][g][str(K)][metric] for s in SEEDS]
        return [r["runs"][f"{view}|{k}|{s}"][g][str(K)][metric] for s in SEEDS]

    def vgrp(ds, k, view, g):
        r = R[ds]
        if view in FIXED:
            return [r["fixed"][f"{view}|{s}"]["pur_grp10"][g] for s in SEEDS]
        if view == "rand1":
            return [r["rand1"][str(s)]["pur_grp10"][g] for s in SEEDS]
        return [r["runs"][f"{view}|{k}|{s}"]["pur_grp10"][g] for s in SEEDS]

    VIEWS = ["X", "AX", "A2X", "rand1", "원본", "대조"]
    VN = {"X": "X", "AX": "ÂX", "A2X": "Â²X", "rand1": "rand1", "원본": "원본", "대조": "대조"}
    L = ["# T19a 요약 — E1: 유사도 기반 pseudo-class 순도 (학습 없음)", ""]
    L.append("- 관점: X, ÂX, Â²X(학습 없음), rand1(seed 0–4), 원본 = T14 `base`, 대조 = T14 `uni`의 `emb_best.npy`. 행 L2 정규화 후 코사인, 자기 자신 제외.")
    L.append("- 동점 규칙: GPU float64 유사도 → 상위 후보를 CPU로 옮겨 (유사도 내림차순, 노드 인덱스 오름차순).")
    L.append("- 그룹 분할은 seed마다 다르다(valid 분할). 학습 없는 관점도 seed별 분할로 계산하여 seed 평균 ± sd로 적는다.")
    L.append("")
    L.append("## T1 — `pur_all` (전체 노드에서 top-K, seed 평균 ± sd)")
    for ds in DATASETS:
        for k in SHOTS:
            L.append(f"### {ds}, {k}-shot")
            L.append("| 관점 | " + " | ".join(f"{g} K={K}" for g in GROUPS for K in KS) + " |")
            L.append("|---|" + "---|" * (len(GROUPS) * len(KS)))
            for v in VIEWS:
                L.append(f"| {VN[v]} | " + " | ".join(ms(vals(ds, k, v, g, K))[0] for g in GROUPS for K in KS) + " |")
            L.append("")
    L.append("## T2 — 짝지은 차 (`pur_all`, seed 단위, 평균 ± SE, 부호 일치 수/5)")
    pairs = [("원본 − ÂX", "원본", "AX"), ("대조 − ÂX", "대조", "AX"), ("원본 − rand1", "원본", "rand1"), ("대조 − 원본", "대조", "원본")]
    for K in (10, 50):
        L.append(f"### K = {K}")
        L.append("| 데이터셋 | shot | " + " | ".join(f"{p[0]} {g}" for p in pairs for g in GROUPS) + " |")
        L.append("|---|---|" + "---|" * (len(pairs) * 3))
        for ds in DATASETS:
            for k in SHOTS:
                L.append(f"| {ds} | {k} | " + " | ".join(paired(vals(ds, k, a, g, K), vals(ds, k, b, g, K))[0] for _, a, b in pairs for g in GROUPS) + " |")
        L.append("")
    L.append("## T3 — `pur_grp` (그룹 안 top-10, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 관점 | base | valid | test |")
    L.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        for k in SHOTS:
            for v in VIEWS:
                L.append(f"| {ds} | {k} | {VN[v]} | " + " | ".join(ms(vgrp(ds, k, v, g))[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T4 — `frac_other`, `frac_unassigned`, `frac_tie` (K = 10, 50, seed 평균)")
    L.append("| 데이터셋 | shot | 관점 | 그룹 | frac_other 10 | frac_other 50 | frac_unassigned 10 | frac_unassigned 50 | frac_tie 10 | frac_tie 50 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for k in SHOTS:
            for v in VIEWS:
                for g in GROUPS:
                    c = [f"{np.mean(vals(ds, k, v, g, K, m)):.4f}" for m in ("frac_other", "frac_unassigned", "frac_tie") for K in (10, 50)]
                    L.append(f"| {ds} | {k} | {VN[v]} | {g} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} |")
    L.append("")
    Q = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "Q5_Q1": 0, "Q5_Q3": 0}
    detail = []
    for ds in DATASETS:
        for k in SHOTS:
            mo = lambda v, g: float(np.mean(vals(ds, k, v, g, 10)))
            Q["Q1"] += mo("원본", "test") < mo("AX", "test")
            Q["Q2"] += mo("원본", "base") > mo("AX", "base")
            _, m, se = paired(vals(ds, k, "대조", "test", 10), vals(ds, k, "원본", "test", 10))
            Q["Q3"] += m > 0 and m > 2 * se
            Q["Q5_Q1"] += mo("원본", "valid") < mo("AX", "valid")
            _, m2, se2 = paired(vals(ds, k, "대조", "valid", 10), vals(ds, k, "원본", "valid", 10))
            Q["Q5_Q3"] += m2 > 0 and m2 > 2 * se2
    for ds in DATASETS:
        mo = lambda v: float(np.mean(vals(ds, 1, v, "test", 10)))
        ok = mo("A2X") >= mo("AX") >= mo("X")
        Q["Q4"] += ok
        detail.append((ds, mo("X"), mo("AX"), mo("A2X"), ok))
    L.append("## 판정 (값만)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| Q1 | test `pur_all`(K = 10) 원본 < ÂX (seed 평균) | {Q['Q1']}/6 |")
    L.append(f"| Q2 | base `pur_all`(K = 10) 원본 > ÂX | {Q['Q2']}/6 |")
    L.append(f"| Q3 | test `pur_all`(K = 10) 대조 − 원본 분명히 양수 | {Q['Q3']}/6 |")
    L.append(f"| Q4 | test `pur_all`(K = 10) Â²X ≥ ÂX ≥ X인 데이터셋 | {Q['Q4']}/3 |")
    L.append(f"| Q5 (기록) | valid 그룹 Q1 / Q3 | {Q['Q5_Q1']}/6 / {Q['Q5_Q3']}/6 |")
    L.append("")
    L.append("- Q4 값(test, K = 10, seed 평균; X / ÂX / Â²X): " + "; ".join(f"{d} {a:.4f} / {b:.4f} / {c:.4f}" for d, a, b, c, _ in detail) + ".")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    yn = lambda b: "일치" if b else "어긋남"
    L.append(f"| P100 | Q1 ≥ 4/6 | {Q['Q1']}/6 | {yn(Q['Q1'] >= 4)} |")
    L.append(f"| P101 | Q2 ≥ 5/6 | {Q['Q2']}/6 | {yn(Q['Q2'] >= 5)} |")
    L.append(f"| P102 | Q3 ≥ 5/6 | {Q['Q3']}/6 | {yn(Q['Q3'] >= 5)} |")
    L.append(f"| P103 | Q4 ≥ 2/3 | {Q['Q4']}/3 | {yn(Q['Q4'] >= 2)} |")
    open(os.path.join(ROOT, "reports", "T19a_summary.md"), "w").write("\n".join(L) + "\n")
    # gates Ga, Ge
    ga = [x for ds in DATASETS for x in R[ds]["ga"]]
    bad = [x for x in ga if not (x["labels_equal"] and x["lists_equal"])]

    def allfinite(o):
        if isinstance(o, dict):
            return all(allfinite(v) for v in o.values())
        if isinstance(o, list):
            return all(allfinite(v) for v in o)
        if isinstance(o, float):
            return math.isfinite(o)
        return True

    ge = {ds: allfinite(R[ds]) for ds in DATASETS}
    short = {ds: sum(v["margin_short"] for part in ("fixed", "rand1", "runs") for v in R[ds][part].values()) for ds in DATASETS}
    gates = {"Ga": {"n_checked": len(ga), "n_bad": len(bad), "bad": bad, "pass": not bad}, "Gb": GBC["Gb"], "Gc_a": GBC["Gc_a"], "Gc_b": GBC["Gc_b"],
             "Ge": ge, "margin_short": short, "Q": Q, "time": {ds: R[ds]["time"] for ds in DATASETS},
             "shapes": sorted({f"{ds}:{tuple(v['shape'])}" for ds in DATASETS for v in R[ds]["runs"].values()})}
    json.dump(gates, open(os.path.join(OUT19A, "gate_checks.json"), "w"), indent=1)
    print(json.dumps(gates, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "t19a-gates":
        t19a_gates()
    elif cmd == "t19a-run":
        t19a_run(sys.argv[2])
    elif cmd == "t19a-report":
        t19a_report()
