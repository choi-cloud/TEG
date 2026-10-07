"""T13 analysis (dimensional collapse diagnostic; analysis only) -> reports/T13_summary.md, reports/T13_spectrum.png,
results/T13_gate_checks.txt.

Spaces: T09 emb_best.npy (base / kl_h2 / infonce; selected lambda from reports/T09_summary.md table 2, plus all 7 configs for
table 4) and the training-free 64-d controls rand1 = A_hat X R (analyze_t06b.rand_spaces) and pca1 = PCA-64 of A_hat X
(A_hat X from analyze_t06b.dataset_spaces; PCA fitted on all nodes, centered, randomized SVD, random_state = 2000 + seed).
Groups: split.json class lists + labels.npy of each run, as in tools/analyze_t12.py main() (table 2). Labels used for metrics only.
Metrics on row-L2-normalized embeddings: erank_node, erank_cls (with C_G), BW, normalized spectrum.
"""
import json
import math
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from analyze_t06b import dataset_spaces, l2n, rand_spaces  # noqa: E402
from analyze_t12 import table_rows  # noqa: E402

ROOT = os.path.dirname(TOOLS)
R09 = os.path.join(ROOT, "results", "T09")
OUT = os.path.join(ROOT, "reports", "T13_summary.md")
FIG = os.path.join(ROOT, "reports", "T13_spectrum.png")
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
GROUPS = ["base", "valid", "test"]
VAR = ["원본", "보존", "대조"]
SPACES = VAR + ["rand1", "pca1"]
CONFIGS7 = ["base", "kl_h2_l0.1", "kl_h2_l1", "kl_h2_l10", "infonce_l0.1", "infonce_l1", "infonce_l10"]
METRICS = ["erank_node", "erank_cls", "BW"]


def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def erank(M):
    """Roy & Vetterli (2007) effective rank of the row-centered matrix M."""
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def erank_node(Z):
    return erank(l2n(Z.astype(np.float64)))


def group_metrics(Z, nodes, y):
    """Z: full embedding; nodes: group node indices; y: their labels."""
    Zh = l2n(Z[nodes].astype(np.float64))
    classes = np.unique(y)
    s = np.linalg.svd(Zh - Zh.mean(0, keepdims=True), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    m_all = Zh.mean(0)
    means = np.stack([Zh[y == c].mean(0) for c in classes])
    n_c = np.array([(y == c).sum() for c in classes])
    S_B = float((n_c * ((means - m_all) ** 2).sum(1)).sum())
    S_W = float(sum(((Zh[y == c] - means[i]) ** 2).sum() for i, c in enumerate(classes)))
    spec = np.zeros(64)
    v = s ** 2 / (s ** 2).sum()
    spec[:min(64, len(v))] = v[:64]
    return {"erank_node": float(np.exp(-(p * np.log(p)).sum())), "erank_cls": erank(l2n(means)), "BW": S_B / S_W,
            "C": len(classes), "n": len(nodes), "spec": spec}


def groups_of(run_dir):
    """Same group construction as tools/analyze_t12.py main() table 2: split.json class lists + labels.npy."""
    sp = json.load(open(os.path.join(run_dir, "split.json")))
    labels = np.load(os.path.join(run_dir, "labels.npy"))
    cl = {"base": sp["class_list_train"], "valid": sp["class_list_valid"], "test": sp["class_list_test"]}
    out = {}
    for g in GROUPS:
        nodes = np.concatenate([np.where(labels == int(c))[0] for c in cl[g]])
        out[g] = (nodes, labels[nodes])
    return out, sp, labels


def main():
    # ---- G3: implementation check
    rng = np.random.default_rng(13000)
    g3_iso = erank_node(rng.standard_normal((5000, 64)))
    g3_r5 = erank_node(rng.standard_normal((5000, 5)) @ rng.standard_normal((5, 64)))

    sel09 = {}
    for r in table_rows(os.path.join(ROOT, "reports", "T09_summary.md"), "## 표 2"):
        if r[2] in ("kl_h2", "infonce"):
            sel09.setdefault((r[0], int(r[1])), {"원본": "base"})["보존" if r[2] == "kl_h2" else "대조"] = f"{r[2]}_l{r[3]}"

    M = {}   # (ds, k, s, space, g) -> metrics ; spaces include VAR, rand1, pca1 and the 7 configs ("cfg:<name>")
    g4 = []
    for ds in DATASETS:
        base = dataset_spaces(ds)
        for s in SEEDS:
            free = {"rand1": rand_spaces(base, s)["rand1"],
                    "pca1": PCA(n_components=64, svd_solver="randomized", random_state=2000 + s).fit_transform(base["diff1"]).astype(np.float32)}
            for k in SHOTS:
                cfg_dirs = {c: os.path.join(R09, c, f"{ds}_5w{k}s", f"seed{s}") for c in CONFIGS7}
                G, sp0, lab0 = groups_of(cfg_dirs["base"])
                same = all(json.load(open(os.path.join(d, "split.json"))) == sp0 and np.array_equal(np.load(os.path.join(d, "labels.npy")), lab0)
                           for d in cfg_dirs.values())
                spaces = {v: np.load(os.path.join(cfg_dirs[sel09[(ds, k)][v]], "emb_best.npy")) for v in VAR} | free | \
                         {f"cfg:{c}": np.load(os.path.join(d, "emb_best.npy")) for c, d in cfg_dirs.items()}
                for name, Z in spaces.items():
                    assert Z.shape[1] == 64, (name, Z.shape)
                    for g in GROUPS:
                        M[(ds, k, s, name, g)] = group_metrics(Z, *G[g])
                nc = {g: {(M[(ds, k, s, n, g)]["n"], M[(ds, k, s, n, g)]["C"]) for n in spaces} for g in GROUPS}
                g4.append(((ds, k, s), same and all(len(v) == 1 for v in nc.values())))
            print(ds, s, "done", flush=True)
        del base

    L = ["# T13 요약 — 진단 1: 차원 붕괴 (분석 전용)", ""]
    L.append("- 공간: 원본 = T09 `base`, 보존 = T09 `kl_h2`(선택 λ), 대조 = T09 `infonce`(선택 λ)의 `emb_best.npy`(64차원). 선택 λ는 `reports/T09_summary.md` 표 2. "
             "rand1 = ÂX·R(R ~ N(0, 1/64), `default_rng(2000 + seed)`), pca1 = ÂX의 PCA 64성분(전체 노드, 중심화, randomized SVD `random_state` = 2000 + seed).")
    L.append("- 모든 지표는 행 L2 정규화 후 계산. seed 0–2 평균 ± sd. n = 그룹 노드 수, C = 그룹 클래스 수.")
    L.append("")

    def nC(ds, k, g):
        ns = sorted({M[(ds, k, s, '원본', g)]['n'] for s in SEEDS})
        Cs = sorted({M[(ds, k, s, '원본', g)]['C'] for s in SEEDS})
        return f"{'/'.join(map(str, ns))}", f"{'/'.join(map(str, Cs))}"

    def ms(ds, k, sp, g, met, fmt="{:.2f}"):
        return mean_sd([M[(ds, k, s, sp, g)][met] for s in SEEDS], fmt)

    titles = {"erank_node": "## 표 1 — erank_node (1–64)", "erank_cls": "## 표 2a — erank_cls (≤ C − 1)", "BW": "## 표 2b — BW = S_B / S_W"}
    for met in METRICS:
        L.append(titles[met])
        for g in GROUPS:
            L.append(f"### 그룹 = {g}")
            L.append("| 데이터셋 | shot | 선택 λ (보존 / 대조) | n (seed 0/1/2) | C | " + " | ".join(SPACES) + " |")
            L.append("|---|---|---|---|---|" + "---|" * len(SPACES))
            for ds, k in CELLS:
                n_, C_ = nC(ds, k, g)
                lam = f"{sel09[(ds, k)]['보존'].rsplit('_l', 1)[1]} / {sel09[(ds, k)]['대조'].rsplit('_l', 1)[1]}"
                fmt = "{:.3f}" if met == "BW" else "{:.2f}"
                L.append(f"| {ds} | {k} | {lam} | {n_} | {C_} | " + " | ".join(ms(ds, k, sp, g, met, fmt)[0] for sp in SPACES) + " |")
            L.append("")

    L.append("## 표 3 — 짝지은 차 (seed 단위, 평균 ± sd, 부호 일치 수/3)")
    J = {"C1": 0, "C2 대조": 0, "C2 보존": 0, "C3": 0, "C4 대조": 0, "C4 보존": 0}
    for met in METRICS:
        fmt = "{:+.3f}" if met == "BW" else "{:+.2f}"
        L.append(f"### {met}")
        L.append("| 데이터셋 | shot | " + " | ".join(f"{v} − 원본 {g}" for v in ("대조", "보존") for g in GROUPS) + " |")
        L.append("|---|---|" + "---|" * 6)
        for ds, k in CELLS:
            cols = []
            for v in ("대조", "보존"):
                for g in GROUPS:
                    d = [M[(ds, k, s, v, g)][met] - M[(ds, k, s, "원본", g)][met] for s in SEEDS]
                    txt, m = mean_sd(d, fmt)
                    sg = np.sign(m)
                    agree = int(sum(np.sign(x) == sg for x in d)) if m != 0 else 0
                    cols.append(f"{txt} ({agree}/3)")
                    if g == "test" and m > 0 and agree == 3:
                        if met == "erank_node":
                            J[f"C2 {v}"] += 1
                        if met == "BW":
                            J[f"C4 {v}"] += 1
            L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
        L.append("")
    for ds, k in CELLS:
        o_t, r_t, p_t = (ms(ds, k, sp, "test", "erank_node")[1] for sp in ("원본", "rand1", "pca1"))
        J["C1"] += o_t < r_t and o_t < p_t
        o_b, p_b = ms(ds, k, "원본", "base", "erank_node")[1], ms(ds, k, "pca1", "base", "erank_node")[1]
        J["C3"] += (o_t / p_t) < (o_b / p_b)

    L.append("## 표 4 — λ에 따른 변화 (test 그룹, T09 7개 설정, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 설정 | erank_node | erank_cls | BW |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for c in CONFIGS7:
            L.append(f"| {ds} | {k} | {c} | {ms(ds, k, 'cfg:' + c, 'test', 'erank_node')[0]} | {ms(ds, k, 'cfg:' + c, 'test', 'erank_cls')[0]} | "
                     f"{ms(ds, k, 'cfg:' + c, 'test', 'BW', '{:.3f}')[0]} |")
    L.append("")

    L.append("## 표 5 — 정확도 병기 (`reports/T12_summary.md` 표 2에서 그대로 옮김, 코사인 프로토타입 probe %)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"{v} {g}" for g in GROUPS for v in VAR) + " |")
    L.append("|---|---|" + "---|" * 9)
    t12 = os.path.join(ROOT, "reports", "T12_summary.md")
    acc = {g: {(r[0], int(r[1])): r[3:6] for r in table_rows(t12, f"### 그룹 = {g}")} for g in GROUPS}
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(x for g in GROUPS for x in acc[g][(ds, k)]) + " |")
    L.append("")

    # ---- figure 1
    fig_note = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, ds in zip(axes, DATASETS):
            for sp in ("원본", "보존", "대조", "pca1"):
                y = np.mean([M[(ds, 1, s, sp, "test")]["spec"] for s in SEEDS], axis=0)
                ax.semilogy(np.arange(1, 65), y, label={"원본": "base", "보존": "kl_h2 (sel)", "대조": "infonce (sel)", "pca1": "pca1"}[sp])
            ax.set_title(f"{ds}, test, 1-shot")
            ax.set_xlabel("k")
        axes[0].set_ylabel("normalized variance σ_k² / Σσ²")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(FIG, dpi=120)
        fig_note = f"`reports/T13_spectrum.png` (matplotlib {matplotlib.__version__})"
    except ImportError:
        fig_note = "생략 — matplotlib 없음"
    L.append(f"## 그림 1 — {fig_note}")
    L.append("- 데이터셋별 패널, test 그룹, 1-shot, seed 평균 정규화 분산(로그 y). 선: 원본(base), 보존(kl_h2 sel), 대조(infonce sel), pca1.")
    L.append("")

    L.append("## 판정 (값만, 단위 (데이터셋, shot) 6개)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| C1 | erank_node(원본, test) < rand1 이고 < pca1 (seed 평균) | {J['C1']}/6 |")
    L.append(f"| C2 | erank_node(test) 차 > 0 분명(seed 3개 모두 양수) | 대조 {J['C2 대조']}/6, 보존 {J['C2 보존']}/6 |")
    L.append(f"| C3 | erank_node 원본/pca1 비율이 test < base (seed 평균) | {J['C3']}/6 |")
    L.append(f"| C4 | BW(test) 차 > 0 분명(seed 3개 모두 양수) | 대조 {J['C4 대조']}/6, 보존 {J['C4 보존']}/6 |")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    yn = lambda b: "일치" if b else "어긋남"
    L.append(f"| P74 | C1 ≥ 4/6 | {J['C1']}/6 | {yn(J['C1'] >= 4)} |")
    L.append(f"| P75 | C2 ≥ 5/6(대조), ≥ 4/6(보존) | 대조 {J['C2 대조']}/6, 보존 {J['C2 보존']}/6 | {yn(J['C2 대조'] >= 5 and J['C2 보존'] >= 4)} |")
    L.append(f"| P76 | C3 ≥ 4/6 | {J['C3']}/6 | {yn(J['C3'] >= 4)} |")
    L.append(f"| P77 | C4 ≥ 5/6(대조), ≥ 5/6(보존) | 대조 {J['C4 대조']}/6, 보존 {J['C4 보존']}/6 | {yn(J['C4 대조'] >= 5 and J['C4 보존'] >= 5)} |")
    open(OUT, "w").write("\n".join(L) + "\n")

    split_across_shots = all(json.load(open(os.path.join(R09, "base", f"{ds}_5w1s", f"seed{s}", "split.json"))) ==
                             json.load(open(os.path.join(R09, "base", f"{ds}_5w5s", f"seed{s}", "split.json"))) for ds in DATASETS for s in SEEDS)
    Gt = [f"G3 isotropic gaussian erank_node = {g3_iso:.4f} (>= 60: {g3_iso >= 60}); rank-5 erank_node = {g3_r5:.4f} (<= 5.5: {g3_r5 <= 5.5})",
          f"G4 per (ds, shot, seed): split/labels identical across 7 configs and n, C_G identical across all spaces: {sum(x[1] for x in g4)}/{len(g4)}; "
          f"failures {[x[0] for x in g4 if not x[1]] or 'none'}",
          f"split identical across shots (base, per ds/seed): {split_across_shots}",
          f"sel09 {sel09}", f"J {J}"]
    open(os.path.join(ROOT, "results", "T13_gate_checks.txt"), "w").write("\n".join(Gt) + "\n")
    print("\n".join(Gt))


if __name__ == "__main__":
    main()
