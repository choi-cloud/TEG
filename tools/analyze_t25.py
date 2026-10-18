"""T25 analysis (no training) -> results/T25/analysis_<ds>.json, results/T25/analysis.json, reports/T25_summary.md.

Representations: S0 = row-normalized A_hat^2 X (fusion_baseline.diffusion_features_hops(ds, 2)); P64 / P16 = PCA scores of S0
(sklearn PCA(n_components=64, svd_solver="randomized", random_state=18500) fitted once per dataset on all nodes without labels,
mean-centred; P16 = first 16 columns of the same fit). S3 (TEG, 64) and G (GPN, 16) are read from results/T21 and results/T24
analysis JSONs (not recomputed).
Probe: analyze_t06b.episodes + fewshot_acc as in tools/analyze_t21.py / analyze_t24.py (episodes from random.Random(1000 + seed),
drawn once per group base -> valid -> test per (ds, k, seed), shared by all V; no pre-normalization); euclidean prototype probe
(analyze_t24.fewshot_acc_euc) for P16 / P64. Purity: analyze_t14.knn_purity (top-10 cosine), S0 / P64 / P16.
Split / labels: results/T22/base split.json / labels.npy.
Usage  python tools/analyze_t25.py run <ds>   -> results/T25/analysis_<ds>.json
       python tools/analyze_t25.py report     -> gates, tables
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from analyze_t24 import fewshot_acc_euc  # noqa: E402  (first: analyze_t24 -> run_t24 sets the tools/gpn import path)

sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fusion_baseline as fb  # noqa: E402
from analyze_t06b import episodes, fewshot_acc, mean_sd, mean_se, sign_agree  # noqa: E402
from analyze_t14 import knn_purity  # noqa: E402

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
GROUPS = ["base", "valid", "test"]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
R21 = os.path.join(ROOT, "results", "T21")
R22 = os.path.join(ROOT, "results", "T22", "base")
R24 = os.path.join(ROOT, "results", "T24")
R25 = os.path.join(ROOT, "results", "T25")
PCA_SEED = 18500
NEW = ["S0", "P64", "P16"]
TABLE = ["S0", "P64", "S3", "P16", "G"]


def pca(S0, n=64):
    """PCA scores (64, 16) of S0 on all nodes, explained variance ratios, orthogonality of C = components_.T (d x 64)."""
    from sklearn.decomposition import PCA

    p = PCA(n_components=n, svd_solver="randomized", random_state=PCA_SEED).fit(S0)
    P64 = p.transform(S0).astype(np.float32)
    C = p.components_.T.astype(np.float64)
    evr = p.explained_variance_ratio_.astype(np.float64)
    info = {"evr": evr.tolist(), "cum16": float(evr[:16].sum()), "cum64": float(evr.sum()),
            "orth_absmax": float(np.abs(C.T @ C - np.eye(n)).max())}
    return P64, np.ascontiguousarray(P64[:, :16]), info


def run(ds):
    torch.set_num_threads(16)
    os.makedirs(R25, exist_ok=True)
    S0 = fb.diffusion_features_hops(ds, 2)
    P64, P16, info = pca(S0)
    spaces = {"S0": S0, "P64": P64, "P16": P16}
    out = {"dim": int(S0.shape[1]), "pca": info, "acc": {}, "euc": {}, "pur": {}, "finite": {n: bool(np.isfinite(V).all()) for n, V in spaces.items()}}
    for s in SEEDS:
        for k in SHOTS:
            ref = os.path.join(R22, f"{ds}_5w{k}s", f"seed{s}")
            sp = json.load(open(os.path.join(ref, "split.json")))
            labels = np.load(os.path.join(ref, "labels.npy"))
            groups = {"base": sp["class_list_train"], "valid": sp["class_list_valid"], "test": sp["class_list_test"]}
            by_class = {c: np.where(labels == int(c))[0].tolist() for g in GROUPS for c in groups[g]}
            rng = random.Random(1000 + s)
            for g in GROUPS:
                eps = episodes(rng, groups[g], by_class, k)
                for n, V in spaces.items():
                    out["acc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc(V, eps, k)
                    if n != "S0":
                        out["euc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc_euc(V, eps, k)
            if k == 5:
                for g in GROUPS:
                    nodes = np.where(np.isin(labels, [int(c) for c in groups[g]]))[0]
                    for n, V in spaces.items():
                        out["pur"][f"{n}|{s}|{g}"] = knn_purity(V, nodes, labels[nodes])
            print(ds, s, k, "done", flush=True)
    json.dump(out, open(os.path.join(R25, f"analysis_{ds}.json"), "w"), indent=1)


def report():
    A = {ds: json.load(open(os.path.join(R25, f"analysis_{ds}.json"))) for ds in DATASETS}
    A21 = {ds: json.load(open(os.path.join(R21, f"analysis_{ds}.json"))) for ds in DATASETS}
    A24 = {ds: json.load(open(os.path.join(R24, f"analysis_{ds}.json"))) for ds in DATASETS}

    def acc(ds, n, k, s, g):
        src = A21 if n == "S3" else A24 if n == "G" else A
        return src[ds]["acc"][f"{n}|{k}|{s}|{g}"]

    def pur(ds, n, s, g):
        src = A21 if n == "S3" else A24 if n == "G" else A
        return src[ds]["pur"][f"{n}|{s}|{g}"]

    def m(xs):
        return float(np.mean(xs))

    def sm(ds, n, k, g):
        return m([acc(ds, n, k, s, g) for s in SEEDS])

    # ---- gates
    G = {}
    G["Ib"] = {f"{ds}|{k}|{s}|{g}": abs(A[ds]["acc"][f"S0|{k}|{s}|{g}"] - A21[ds]["acc"][f"S0|{k}|{s}|{g}"])
               for ds, k in CELLS for s in SEEDS for g in GROUPS}
    G["Ic"] = {}
    for ds in DATASETS:
        evr = np.array(A[ds]["pca"]["evr"])
        cum = np.cumsum(evr)
        G["Ic"][ds] = {"orth_absmax": A[ds]["pca"]["orth_absmax"], "evr_finite": bool(np.isfinite(evr).all()),
                       "cum_increasing": bool(np.all(np.diff(cum) > 0)), "cum16": A[ds]["pca"]["cum16"], "cum64": A[ds]["pca"]["cum64"]}
    vals = [x for ds in DATASETS for key in ("acc", "euc", "pur") for x in A[ds][key].values()]
    G["Ie"] = {"all_metrics_finite": bool(np.isfinite(vals).all()), "n_metrics": len(vals),
               "all_spaces_finite": all(all(A[ds]["finite"].values()) for ds in DATASETS)}
    G["pass"] = {"Ib": max(G["Ib"].values()) <= 1e-6, "Ib_max": max(G["Ib"].values()),
                 "Ic": all(v["orth_absmax"] <= 1e-4 and v["evr_finite"] and v["cum_increasing"] for v in G["Ic"].values()),
                 "Ie": G["Ie"]["all_metrics_finite"] and G["Ie"]["all_spaces_finite"]}

    # ---- judgements (values only)
    J = {}
    J["K1"] = sum(sm(ds, "G", k, "test") < sm(ds, "P16", k, "test") for ds, k in CELLS)
    J["K2"] = sum(sm(ds, "G", k, "base") > sm(ds, "P16", k, "base") for ds, k in CELLS)
    J["K3"] = sum(sm(ds, "S3", k, "test") < sm(ds, "P64", k, "test") for ds, k in CELLS)
    J["K4"] = sum(sm(ds, "S3", k, "base") > sm(ds, "P64", k, "base") for ds, k in CELLS)
    J["K5_G"] = sum(sm(ds, "G", k, "valid") < sm(ds, "P16", k, "valid") for ds, k in CELLS)
    J["K5_S3"] = sum(sm(ds, "S3", k, "valid") < sm(ds, "P64", k, "valid") for ds, k in CELLS)
    J["K6_P16"] = sum(sm(ds, "P16", k, "test") < sm(ds, "S0", k, "test") for ds, k in CELLS)
    J["K6_P64"] = sum(sm(ds, "P64", k, "test") < sm(ds, "S0", k, "test") for ds, k in CELLS)
    P = {"P131": ("K1 ≥ 4/6", f"K1 = {J['K1']}/6"), "P132": ("K2 ≥ 5/6", f"K2 = {J['K2']}/6"),
         "P133": ("K3 ≤ 3/6", f"K3 = {J['K3']}/6"), "P134": ("K4 ≥ 5/6", f"K4 = {J['K4']}/6")}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return f"{mean_se(d)[0]} ({sign_agree(d)}/3)"

    L = ["# T25 요약 — Â²X의 PCA(16·64) vs 학습한 임베딩 (학습 없음)", ""]
    L.append("- S0 = 행 정규화 Â²X. P64 / P16 = S0의 PCA 64 / 16차원 점수(`PCA(n_components=64, svd_solver=\"randomized\", "
             "random_state=18500)`를 전체 노드에 레이블 없이 한 번 적합, 평균 중심화, P16은 앞 16개 성분). S3 = TEG 원본 GCN(64, "
             "`results/T21/analysis_<ds>.json` 전재), G = GPN(16, `results/T24/analysis_<ds>.json` 전재). seed 0–2.")
    L.append("- probe: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid "
             "→ test 순, 모든 표현이 공유, 사전 정규화 없음). 정확도 %. 분할 = T22 `base` split.json·labels.npy.")
    L.append("")
    L.append("## T1 — (seed 평균 ± sd; S3는 T21, G는 T24 전재)")
    L.append("| 데이터셋 | k | 표현 | acc_base | acc_valid | acc_test |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for n in TABLE:
            L.append(f"| {ds} | {k} | {n} | " + " | ".join(mean_sd([acc(ds, n, k, s, g) for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T2 — seed 단위 짝지은 차 (%p, 평균 ± SE, 부호 일치/3)")
    hdr = [("test", "G", "P16"), ("valid", "G", "P16"), ("base", "G", "P16"), ("test", "S3", "P64"), ("valid", "S3", "P64"),
           ("base", "S3", "P64"), ("test", "P16", "S0"), ("test", "P64", "S0")]
    L.append("| 데이터셋 | k | " + " | ".join(f"{g}: {a} − {b}" for g, a, b in hdr) + " |")
    L.append("|---|---|" + "---|" * len(hdr))
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(paired([acc(ds, a, k, s, g) for s in SEEDS], [acc(ds, b, k, s, g) for s in SEEDS])
                                                  for g, a, b in hdr) + " |")
    L.append("")
    L.append("## T3 — 순도 (top-10 코사인 kNN, seed 평균 ± sd; S3는 T21, G는 T24 전재, 둘 다 5-shot run)")
    L.append("| 데이터셋 | 표현 | pur_base | pur_valid | pur_test |")
    L.append("|---|---|---|---|---|")
    for ds in DATASETS:
        for n in ("S0", "P64", "P16", "S3", "G"):
            L.append(f"| {ds} | {n} | " + " | ".join(mean_sd([pur(ds, n, s, g) for s in SEEDS], "{:.4f}")[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T4 — 설명 분산 비율과 유클리드 probe (기록)")
    L.append("| 데이터셋 | S0 차원 | 누적 설명 분산 16 | 누적 설명 분산 64 | \\|CᵀC − I\\|max |")
    L.append("|---|---|---|---|---|")
    for ds in DATASETS:
        p = A[ds]["pca"]
        L.append(f"| {ds} | {A[ds].get('dim', '')} | {p['cum16']:.4f} | {p['cum64']:.4f} | {p['orth_absmax']:.2e} |")
    L.append("")
    L.append("| 데이터셋 | k | 표현 | acc_base (euc) | acc_valid (euc) | acc_test (euc) |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for n in ("P64", "P16"):
            L.append(f"| {ds} | {k} | {n} | " + " | ".join(mean_sd([A[ds]["euc"][f"{n}|{k}|{s}|{g}"] for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## 판정 (값만, seed 평균, 코사인 probe)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| K1 | acc_test(G) < acc_test(P16) | {J['K1']}/6 |")
    L.append(f"| K2 | acc_base(G) > acc_base(P16) | {J['K2']}/6 |")
    L.append(f"| K3 | acc_test(S3) < acc_test(P64) | {J['K3']}/6 |")
    L.append(f"| K4 | acc_base(S3) > acc_base(P64) | {J['K4']}/6 |")
    L.append(f"| K5 (기록) | acc_valid(G) < acc_valid(P16); acc_valid(S3) < acc_valid(P64) | {J['K5_G']}/6; {J['K5_S3']}/6 |")
    L.append(f"| K6 (기록) | acc_test(P16) < acc_test(S0); acc_test(P64) < acc_test(S0) | {J['K6_P16']}/6; {J['K6_P64']}/6 |")
    L.append("")
    L.append("## 사전 등록 (조건과 계산값만)")
    L.append("| 예측 | 조건 | 계산값 |")
    L.append("|---|---|---|")
    for p, (c, obs) in P.items():
        L.append(f"| {p} | {c} | {obs} |")
    L.append("")
    open(os.path.join(ROOT, "reports", "T25_summary.md"), "w").write("\n".join(L))
    json.dump({"gates": G, "J": J, "P": P}, open(os.path.join(R25, "analysis.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(G["pass"]), json.dumps(G["Ic"]), json.dumps(J))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()
