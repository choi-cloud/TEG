"""T30 analysis: per-test-class decomposition of the T22 fixed test episodes (no training, no new RNG).

Episodes: T22 `base` fixed_eval_logits.npz, mode 1 (200 per seed, Random(9000 + seed)), ep_idx order, seed 0-9.
Arms: X0 / D = cosine prototype on row-L2 X / row-L2 Â²X (fusion_baseline.view_d_scores argmax); TEG / C_all = logits argmax
of `base` / selected-lambda `infonce_all`. Per class c: accuracy of the 5 queries of class c in every episode containing c.
Overlap (per seed split from results/T29/splits.json, then seed mean):
  e_c  D space: ||U delta_c||^2 / ||delta_c||^2, U = right singular vectors (nonzero singular values) of the centered
       base class-mean matrix, delta_c = mu_c - mean(other test class means)
  r_c  X0 space: mean over the 50 columns with largest delta_c^X of f_j = fraction of base nodes with X_ij != 0
Usage  python tools/analyze_t30.py run <ds>   -> results/T30/classwise_<ds>.json
       python tools/analyze_t30.py report     -> gates, reports/T30_summary.md, reports/T30_per_class.md, results/T30/gate_checks.txt
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import fusion_baseline as fb  # noqa: E402
from analyze_t22 import paired  # noqa: E402
from analyze_t28 import CELLS, DATASETS, LAM, SEEDS, SHOTS, call_cfg, npz_path  # noqa: E402

N_WAY, TOPC = 5, 50
ARMS = ["X0", "D", "TEG", "C_all"]
PAIRS = [("TEG", "D"), ("C_all", "TEG"), ("C_all", "D"), ("D", "X0")]
R22B = os.path.join(ROOT, "results", "T22", "base")
SPLITS = os.path.join(ROOT, "results", "T29", "splits.json")
R28 = os.path.join(ROOT, "results", "T28")
OUTD = os.path.join(ROOT, "results", "T30")
OUT = os.path.join(ROOT, "reports", "T30_summary.md")
OUT_PC = os.path.join(ROOT, "reports", "T30_per_class.md")


def labels_path(ds):
    return os.path.join(R22B, f"{ds}_5w1s", "seed0", "labels.npy")


def load_eps(npz):
    Z = np.load(npz)
    m = np.where(Z["mode"] == 1)[0]
    m = m[np.argsort(Z["ep_idx"][m], kind="stable")]
    return Z["support"][m], Z["query"][m], Z["query_y"][m], Z["classes"][m], Z["logits"][m]


def per_class_correct(pred, qy):
    """(E, n_way*Q) predictions in class order -> (E, n_way) number of correct queries per class block."""
    E = pred.shape[0]
    return (pred == qy).reshape(E, N_WAY, -1).sum(-1)


def class_means(F, labels, classes):
    return np.stack([F[labels == c].astype(np.float64).mean(0) for c in classes])


def class_overlap(Dm, Xm, X0, labels, sp):
    """One seed: Dm / Xm = {class: mean} in D / X0 space. Returns per-test-class e, r and seed baselines."""
    base, valid, test = sp["base"], sp["valid"], sp["test"]
    B = np.stack([Dm[c] for c in base])
    B = B - B.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    rank = int((s > s.max() * max(B.shape) * np.finfo(np.float64).eps).sum())
    U = Vt[:rank]

    def e_of(group):
        M = np.stack([Dm[c] for c in group])
        out = {}
        for i, c in enumerate(group):
            delta = M[i] - np.delete(M, i, 0).mean(0)
            out[c] = float((U @ delta) @ (U @ delta) / (delta @ delta))
        return out

    f = (X0[np.isin(labels, base)] != 0).mean(0)
    MX = np.stack([Xm[c] for c in test])
    r = {}
    for i, c in enumerate(test):
        delta = MX[i] - np.delete(MX, i, 0).mean(0)
        r[c] = float(f[np.argsort(-delta, kind="stable")[:TOPC]].mean())
    ev = e_of(valid)
    return {"e": e_of(test), "r": r, "rank": rank, "n_base": len(base), "d": int(B.shape[1]),
            "e_rand": rank / B.shape[1], "f_bar": float(f.mean()),
            "e_valid": [float(np.min(list(ev.values()))), float(np.median(list(ev.values()))), float(np.max(list(ev.values())))]}


def run(ds):
    V = {"X0": fb.diffusion_features_hops(ds, 0), "D": fb.diffusion_features(ds)}
    labels = np.load(labels_path(ds))
    S = json.load(open(SPLITS))["splits"][ds]
    test = [int(c) for c in S["test"]]
    all_cls = sorted(int(c) for c in np.unique(labels))
    Dm = dict(zip(all_cls, class_means(V["D"], labels, all_cls)))
    Xm = dict(zip(all_cls, class_means(V["X0"], labels, all_cls)))
    out = {"test": test, "finite_feat": {n: bool(np.isfinite(v).all()) for n, v in V.items()},
           "seed_overlap": {}, "cell": {}, "t28_check": {}}
    for s in SEEDS:
        sp = {g: [int(c) for c in S["seeds"][str(s)][g]] for g in ("base", "valid")}
        sp["test"] = test
        out["seed_overlap"][str(s)] = class_overlap(Dm, Xm, V["X0"], labels, sp)
    print(ds, "overlap done", flush=True)
    t28 = json.load(open(os.path.join(R28, f"analysis_{ds}.json")))["fixed"]
    for k in SHOTS:
        # vals[arm][c][s] = list of per-episode accuracies (%) of class c's queries
        vals = {a: {c: {s: [] for s in SEEDS} for c in test} for a in ARMS}
        tot = {a: [0, 0] for a in ARMS}
        t28d = {a: 0.0 for a in ARMS}
        for s in SEEDS:
            sup, qry, qy, cls, _ = load_eps(npz_path("base", ds, k, s))
            lb = labels[qry].reshape(len(qry), N_WAY, -1)
            assert (lb == cls[:, :, None]).all() and (labels[sup].reshape(len(sup), N_WAY, -1) == cls[:, :, None]).all()
            pred = {n: fb.view_d_scores(V[n], sup, qry, k).argmax(-1) for n in ("X0", "D")}
            for n in ("X0", "D"):
                assert np.array_equal((pred[n] == qy).mean(-1), fb.cosine_probe_acc(V[n], sup, qry, k))
            pred["TEG"] = load_eps(npz_path("base", ds, k, s))[4].argmax(-1)
            _, _, qy2, cls2, lg = load_eps(npz_path(call_cfg(ds, k), ds, k, s))
            assert np.array_equal(qy2, qy) and np.array_equal(cls2, cls)
            pred["C_all"] = lg.argmax(-1)
            nq = qry.shape[1] // N_WAY
            for a in ARMS:
                cc = per_class_correct(pred[a], qy)
                tot[a][0] += int(cc.sum())
                tot[a][1] += int(cc.size * nq)
                t28d[a] = max(t28d[a], float(np.abs((cc.sum(1) / qry.shape[1]) - np.asarray(t28[f"{a}|{k}|{s}"])).max()))
                for e in range(len(cls)):
                    for j in range(N_WAY):
                        vals[a][int(cls[e, j])][s].append(100.0 * cc[e, j] / nq)
        cell = {"overall": {a: 100.0 * tot[a][0] / tot[a][1] for a in ARMS}, "classes": {}}
        out["t28_check"][str(k)] = t28d
        for c in test:
            rec = {"n_ep": sum(len(vals["D"][c][s]) for s in SEEDS)}
            for a in ARMS:
                x = np.concatenate([vals[a][c][s] for s in SEEDS])
                rec[a] = [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))]
            for a, b in PAIRS:
                dd = [np.asarray(vals[a][c][s]) - np.asarray(vals[b][c][s]) for s in SEEDS if len(vals[a][c][s])]
                p = paired(dd)
                rec[f"{a}-{b}"] = [p[0], p[1], p[2]]
            rec["e"] = float(np.mean([out["seed_overlap"][str(s)]["e"][c] for s in SEEDS]))
            rec["r"] = float(np.mean([out["seed_overlap"][str(s)]["r"][c] for s in SEEDS]))
            rec["e_seed"] = [out["seed_overlap"][str(s)]["e"][c] for s in SEEDS]
            rec["r_seed"] = [out["seed_overlap"][str(s)]["r"][c] for s in SEEDS]
            cell["classes"][str(c)] = rec
        out["cell"][str(k)] = cell
        print(ds, k, "classwise done", flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(out, open(os.path.join(OUTD, f"classwise_{ds}.json"), "w"))


def t28_table1():
    """reports/T28_summary.md 표 T1: (ds, k) -> {X0, D, TEG, C_all} seed means as printed."""
    lines = open(os.path.join(ROOT, "reports", "T28_summary.md")).read().split("\n")
    i = lines.index("## T1 — 고정 test 정확도 (seed 평균 ± sd)")
    hdr = [c.strip() for c in lines[i + 1].strip("|").split("|")]
    out = {}
    for ln in lines[i + 3 : i + 9]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        out[(c[0], int(c[1]))] = {n: float(c[hdr.index(n)].split("±")[0]) for n in ARMS}
    return out


def report():
    A = {ds: json.load(open(os.path.join(OUTD, f"classwise_{ds}.json"))) for ds in DATASETS}
    cls_of = lambda ds, k: A[ds]["cell"][str(k)]["classes"]
    clear_pos = lambda p: p[1] > 0 and p[1] > 2 * p[2]
    clear_neg = lambda p: p[1] < 0 and -p[1] > 2 * p[2]
    G = []

    # Ga
    n_base = sum(os.path.exists(npz_path("base", ds, k, s)) for ds, k in CELLS for s in SEEDS)
    n_call = sum(os.path.exists(npz_path(call_cfg(ds, k), ds, k, s)) for ds, k in CELLS for s in SEEDS)
    sp_ok = os.path.exists(SPLITS) and json.load(open(SPLITS))["Gs_pass"]
    n_lab = sum(os.path.exists(labels_path(ds)) for ds in DATASETS)
    G.append(("Ga 선행", f"base npz {n_base}/60, infonce_all(선택 λ) npz {n_call}/60, splits.json 존재·Gs 통과 {sp_ok}, labels.npy {n_lab}/3",
              n_base == 60 and n_call == 60 and sp_ok and n_lab == 3))
    # Gb
    ref = t28_table1()
    gb_rows = [(ds, k, a, A[ds]["cell"][str(k)]["overall"][a], ref[(ds, k)][a]) for ds, k in CELLS for a in ARMS]
    gb_max = max(abs(x - y) for *_, x, y in gb_rows)
    t28_max = max(v for ds in DATASETS for k in SHOTS for v in A[ds]["t28_check"][str(k)].values())
    G.append(("Gb 재현", f"24값 \\|Δ\\| 최대 {gb_max:.4f} (기준 ≤ 0.005, T28 표 T1은 소수 둘째 자리); "
                       f"기록: `results/T28/analysis_<ds>.json` 에피소드 값과의 \\|Δ\\| 최대 {t28_max:.2e}", gb_max <= 0.005))
    # Gc
    low = {(ds, k, c) for ds, k in CELLS for c, r in cls_of(ds, k).items() if r["n_ep"] < 100}
    nep = [r["n_ep"] for ds, k in CELLS for r in cls_of(ds, k).values()]
    G.append(("Gc 표본", f"n_ep 최소 {min(nep)}, 100 미만 클래스 {len(low)}개", not low))
    # Ge
    nums = []
    for ds in DATASETS:
        for so in A[ds]["seed_overlap"].values():
            nums += list(so["e"].values()) + list(so["r"].values()) + so["e_valid"] + [so["e_rand"], so["f_bar"]]
        for k in SHOTS:
            for r in cls_of(ds, k).values():
                nums += [r[a][0] for a in ARMS] + [r[a][1] for a in ARMS] + [r[f"{a}-{b}"][1] for a, b in PAIRS] + [r[f"{a}-{b}"][2] for a, b in PAIRS]
    fin = all(math.isfinite(x) for x in nums) and all(all(A[ds]["finite_feat"].values()) for ds in DATASETS)
    ranks = {ds: sorted({so["rank"] for so in A[ds]["seed_overlap"].values()}) for ds in DATASETS}
    rank_eq = all(so["rank"] == so["n_base"] - 1 for ds in DATASETS for so in A[ds]["seed_overlap"].values())
    G.append(("Ge 유한·rank", f"유한 {fin}; base 부분공간 rank (seed별 값의 집합) " + ", ".join(f"{ds} {ranks[ds]}" for ds in DATASETS)
              + f"; 모든 seed에서 rank = n_base − 1: {rank_eq}", fin))

    fail_stop = not (G[0][2] and G[1][2])
    with open(os.path.join(OUTD, "gate_checks.txt"), "w") as f:
        for g, v, ok in G:
            f.write(f"{g}: {v} -> {'PASS' if ok else 'FAIL'}\n")
    for g, v, ok in G:
        print(g, v, "PASS" if ok else "FAIL")
    if fail_stop:
        print("Ga/Gb FAIL -> stop")
        return

    fmt_a = lambda v: f"{v[0]:.2f} ± {v[1]:.2f}"
    mark = lambda ds, k, c: f"{c}*" if (ds, k, c) in low else str(c)

    # per-class report
    P = ["# T30 클래스별 전체 표 — T22 고정 test 에피소드 (seed 0–9)", ""]
    P.append("- 정렬: TEG − D 평균 오름차순. 정확도 %: 그 클래스의 query 5개 정확도의 에피소드 평균 ± SE. "
             "짝지은 차 %p: 평균 ± SE (양수 비율, seed 부호 일치/에피소드가 있는 seed 수).")
    P.append("- e_c, r_c: seed 0–9 평균. `*` = n_ep < 100 (Gc).")
    P.append("")
    for ds, k in CELLS:
        C = cls_of(ds, k)
        P.append(f"## {ds} {k}-shot")
        P.append("| 클래스 | n_ep | X0 | D | TEG | C_all | " + " | ".join(f"{a} − {b}" for a, b in PAIRS) + " | e_c | r_c |")
        P.append("|---|---|---|---|---|---|" + "---|" * len(PAIRS) + "---|---|")
        for c in sorted(C, key=lambda c: C[c]["TEG-D"][1]):
            r = C[c]
            P.append(f"| {mark(ds, k, c)} | {r['n_ep']} | " + " | ".join(fmt_a(r[a]) for a in ARMS) + " | "
                     + " | ".join(r[f"{a}-{b}"][0] for a, b in PAIRS) + f" | {r['e']:.4f} | {r['r']:.4f} |")
        P.append("")
    open(OUT_PC, "w").write("\n".join(P) + "\n")

    L = ["# T30 요약 — 클래스별 분해 (학습 없음), T22 고정 test 에피소드", ""]
    L.append("- 에피소드: T22 `base` `fixed_eval_logits.npz` mode 1 (seed당 200, `Random(9000 + seed)`), `ep_idx` 순, seed 0–9.")
    L.append("- arm: X0 / D = `fusion_baseline.view_d_scores` argmax (행 L2 X / Â²X 코사인 프로토타입), TEG / C_all = `logits` argmax "
             "(`base` / 선택 λ `infonce_all`: " + ", ".join(f"{ds} {k}-shot {LAM[(ds, k)]}" for ds, k in CELLS) + ").")
    L.append("- 클래스별 정확도: 그 클래스가 포함된 에피소드마다 그 클래스 query 5개의 정확도. 짝지은 차는 같은 에피소드끼리, "
             "분명 = 평균 > 0이고 > 2·SE (음수는 부호 반대).")
    L.append("- e_c: D 공간, 분할은 `results/T29/splits.json` (seed별), base 클래스 평균 중심화 행렬의 SVD 오른쪽 특이벡터(특이값 > s_max·max(shape)·eps). "
             f"r_c: X0 공간, δ 상위 {TOPC}열의 base 노드 비영 비율 평균. 둘 다 seed 평균.")
    L.append("- 3분위: 값 ≤ `np.quantile(·, 1/3)`인 클래스를 하위 3분위로 둔다. Spearman: `scipy.stats.spearmanr`, 단위 = 그 칸의 test 클래스 전체.")
    L.append("- 클래스별 전체 표: `reports/T30_per_class.md`.")
    L.append("")

    # T1
    L.append("## T1 — 클래스 수와 짝지은 차의 부호")
    L.append("| 데이터셋 | shot | test 클래스 수 | n_ep 최소 | n_ep 중앙 | TEG − D 분명한 음수 | 분명한 양수 | 그 외 | C_all − TEG 분명한 양수 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    T1 = {}
    for ds, k in CELLS:
        C = cls_of(ds, k)
        ne = [r["n_ep"] for r in C.values()]
        neg = sum(clear_neg(r["TEG-D"]) for r in C.values())
        pos = sum(clear_pos(r["TEG-D"]) for r in C.values())
        cpos = sum(clear_pos(r["C_all-TEG"]) for r in C.values())
        T1[(ds, k)] = (neg, pos)
        L.append(f"| {ds} | {k} | {len(C)} | {min(ne)} | {np.median(ne):g} | {neg} | {pos} | {len(C) - neg - pos} | {cpos} |")
    L.append("")

    # T2
    RHO = [("e_c", "TEG − D", lambda r: r["e"], lambda r: r["TEG-D"][1]),
           ("TEG − D", "C_all − TEG", lambda r: r["TEG-D"][1], lambda r: r["C_all-TEG"][1]),
           ("TEG − D", "C_all − D", lambda r: r["TEG-D"][1], lambda r: r["C_all-D"][1]),
           ("r_c", "TEG − D", lambda r: r["r"], lambda r: r["TEG-D"][1]),
           ("e_c", "acc_D", lambda r: r["e"], lambda r: r["D"][0]),
           ("r_c", "acc_D", lambda r: r["r"], lambda r: r["D"][0])]
    R = {}
    L.append("## T2 — Spearman ρ (p), 단위 = test 클래스")
    L.append("| 데이터셋 | shot | " + " | ".join(f"ρ({a}, {b})" for a, b, *_ in RHO) + " |")
    L.append("|---|---|" + "---|" * len(RHO))
    for ds, k in CELLS:
        C = list(cls_of(ds, k).values())
        row = []
        for i, (a, b, fa, fb_) in enumerate(RHO):
            rho, p = spearmanr([fa(r) for r in C], [fb_(r) for r in C])
            R[(ds, k, i)] = float(rho)
            row.append(f"{rho:+.3f} ({p:.3g})")
        L.append(f"| {ds} | {k} | " + " | ".join(row) + " |")
    L.append("")

    # T3
    L.append("## T3 — 3분위 교차표 (e_c 하위 3분위 / 나머지 × acc_D 하위 3분위, TEG − D 분명한 음수)")
    L.append("| 데이터셋 | shot | e_c 하위 수 | (a) acc_D 하위 | (b) acc_D 하위 아님·TEG − D 분명한 음수 | 그 외 | "
             "e_c 나머지 수 | acc_D 하위 | acc_D 하위 아님·TEG − D 분명한 음수 | 그 외 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    T3 = {}
    for ds, k in CELLS:
        C = list(cls_of(ds, k).values())
        qe = np.quantile([r["e"] for r in C], 1 / 3)
        qa = np.quantile([r["D"][0] for r in C], 1 / 3)
        row = []
        for in_low in (True, False):
            grp = [r for r in C if (r["e"] <= qe) == in_low]
            a = sum(r["D"][0] <= qa for r in grp)
            b = sum(r["D"][0] > qa and clear_neg(r["TEG-D"]) for r in grp)
            row += [len(grp), a, b, len(grp) - a - b]
            if in_low:
                T3[(ds, k)] = (a, b)
        L.append(f"| {ds} | {k} | " + " | ".join(str(x) for x in row) + " |")
    L.append("")

    # T4
    L.append("## T4 — TEG − D 최저 3개 / 최고 3개 클래스")
    L.append("| 데이터셋 | shot | 구분 | 클래스 | TEG − D | X0 | D | TEG | C_all | e_c | r_c |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        C = cls_of(ds, k)
        order = sorted(C, key=lambda c: C[c]["TEG-D"][1])
        for tag, cs in (("최저", order[:3]), ("최고", order[-3:][::-1])):
            for c in cs:
                r = C[c]
                L.append(f"| {ds} | {k} | {tag} | {mark(ds, k, c)} | {r['TEG-D'][1]:+.2f} | "
                         + " | ".join(f"{r[a][0]:.2f}" for a in ARMS) + f" | {r['e']:.4f} | {r['r']:.4f} |")
    L.append("")

    # T5
    L.append("## T5 — 기준값 (데이터셋·seed별)")
    L.append("| 데이터셋 | seed | n_base | base 부분공간 rank | d | e_rand = rank/d | f̄ | valid e 최소 | 중앙 | 최대 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for s in SEEDS:
            so = A[ds]["seed_overlap"][str(s)]
            L.append(f"| {ds} | {s} | {so['n_base']} | {so['rank']} | {so['d']} | {so['e_rand']:.4f} | {so['f_bar']:.4f} | "
                     + " | ".join(f"{v:.4f}" for v in so["e_valid"]) + " |")
    L.append("")
    L.append("test 클래스 e_c·r_c 분포 (seed 평균값의 최소 / 중앙 / 최대)")
    L.append("")
    L.append("| 데이터셋 | e_c 최소 | 중앙 | 최대 | r_c 최소 | 중앙 | 최대 |")
    L.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        C = list(cls_of(ds, 1).values())
        e, r = [x["e"] for x in C], [x["r"] for x in C]
        L.append(f"| {ds} | {min(e):.4f} | {np.median(e):.4f} | {max(e):.4f} | {min(r):.4f} | {np.median(r):.4f} | {max(r):.4f} |")
    L.append("")

    # verdicts
    fmt = lambda cs: ", ".join(f"{ds} {k}" for ds, k in cs) or "없음"
    q1 = [c for c in CELLS if T1[c][0] > T1[c][1]]
    q2 = [c for c in CELLS if R[(*c, 0)] > 0]
    q3 = [c for c in CELLS if R[(*c, 1)] < 0]
    q4a = [c for c in CELLS if R[(*c, 2)] < 0]
    q4b = [c for c in CELLS if R[(*c, 3)] > 0]
    L.append("## 판정 (값만, 단위 (ds, k) 6개)")
    L.append("| 판정 | 정의 | 값 | 해당 칸 |")
    L.append("|---|---|---|---|")
    L.append(f"| Q1 | TEG − D 분명한 음수 클래스 수 > 분명한 양수 클래스 수 | {len(q1)}/6 | {fmt(q1)} |")
    L.append(f"| Q2 | ρ(e_c, TEG − D) > 0 | {len(q2)}/6 | {fmt(q2)} |")
    L.append(f"| Q3 | ρ(TEG − D, C_all − TEG) < 0 | {len(q3)}/6 | {fmt(q3)} |")
    L.append(f"| Q4 (기록) | ρ(TEG − D, C_all − D) < 0 | {len(q4a)}/6 | {fmt(q4a)} |")
    L.append(f"| Q4 (기록) | ρ(r_c, TEG − D) > 0 | {len(q4b)}/6 | {fmt(q4b)} |")
    L.append("| Q4 (기록) | T3 (a), (b) 칸별 수 | " + "; ".join(f"{ds} {k}: ({T3[(ds, k)][0]}, {T3[(ds, k)][1]})" for ds, k in CELLS) + " | |")
    L.append("")
    nq1 = [c for c in CELLS if c not in q1]
    L.append("## 사전 등록 계산값")
    L.append("| 예측 | 내용 | 계산값 |")
    L.append("|---|---|---|")
    L.append(f"| P152 | Q1 = 5/6이고, 아닌 칸은 dblp 1-shot | Q1 = {len(q1)}/6; 아닌 칸: {fmt(nq1)} |")
    L.append(f"| P153 | Q2 ≥ 4/6 | Q2 = {len(q2)}/6 |")
    L.append(f"| P154 | Q3 = 6/6 | Q3 = {len(q3)}/6 |")
    L.append("")
    L.append("## 게이트")
    L.append("| 게이트 | 값 | 충족 |")
    L.append("|---|---|---|")
    for g, v, ok in G:
        L.append(f"| {g} | {v} | {'예' if ok else '아니오'} |")
    L.append("| Gd 변경 범위 | 보고서 커밋 뒤 부기에 기록 | — |")
    L.append("")
    L.append("### Gb 상세 (query 수 가중 합산, %)")
    L.append("| 데이터셋 | shot | 열 | T30 | T28 표 T1 | \\|Δ\\| |")
    L.append("|---|---|---|---|---|---|")
    for ds, k, a, x, y in gb_rows:
        L.append(f"| {ds} | {k} | {a} | {x:.4f} | {y:.2f} | {abs(x - y):.4f} |")
    open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "report":
        report()
