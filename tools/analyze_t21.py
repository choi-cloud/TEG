"""T21 analysis -> results/T21/analysis.json, reports/T21_summary.md, reports/T21_ladder.png.

Representations V: S0 = row-normalized A_hat^2 X (fusion_baseline.diffusion_features_hops(ds, 2)); S1 epoch e =
results/T21/mlp/<ds>/seed<s>/h_ep<e>.npy; S2 = results/T21/gamma0 emb_best.npy; S3 = T09 `base` emb_best.npy;
대조 = T09 `infonce_l10` emb_best.npy. S2 / S3 / 대조 use the run of the probed shot; S0 / S1 are shared by both k.
Probe: analyze_t06b.episodes + fewshot_acc as in tools/analyze_t12.py (episodes from random.Random(1000 + seed), drawn once
per group base -> valid -> test per (ds, k, seed), shared by all V; vectors are passed without pre-normalization).
LR head: per S0 test episode, sklearn LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000) fit on the support only.
Purity: analyze_t14.knn_purity(V, nodes_g, y) (top-10 cosine). S2 / S3 / 대조 purity uses the 5-shot run.
Split / labels: T09 `base` split.json / labels.npy (Hb checks T09 infonce_l10 and T21 gamma0 against them).
Usage  python tools/analyze_t21.py run <ds>   -> results/T21/analysis_<ds>.json
       python tools/analyze_t21.py report     -> gates, tables, figure
"""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fusion_baseline as fb  # noqa: E402
from analyze_t06b import episodes, fewshot_acc, mean_sd, mean_se, sign_agree  # noqa: E402  (analyze_t06b chdirs to ROOT)
from analyze_t14 import knn_purity  # noqa: E402

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
GROUPS = ["base", "valid", "test"]
EPOCHS = [0, 25, 50, 100, 200, 500, 1000]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
R09 = os.path.join(ROOT, "results", "T09")
R21 = os.path.join(ROOT, "results", "T21")
RUNS = {"S2": os.path.join(R21, "gamma0"), "S3": os.path.join(R09, "base"), "대조": os.path.join(R09, "infonce_l10")}
LADDER = ["S0", "S1 ep200", "S1 ep1000", "S2", "S3", "대조"]  # S1 ep1000: record column (T21 Hc(1) addendum)


def rdir(v, ds, k, s):
    return os.path.join(RUNS[v], f"{ds}_5w{k}s", f"seed{s}")


def lr_acc(V, eps, k):
    from sklearn.linear_model import LogisticRegression

    accs = []
    ys = np.repeat(np.arange(5), k)
    yq = np.repeat(np.arange(5), 5)
    for sup, qry in eps:
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(V[sup], ys)
        accs.append((clf.predict(V[qry]) == yq).mean())
    return 100 * float(np.mean(accs))


def run(ds):
    torch.set_num_threads(16)
    S0 = fb.diffusion_features_hops(ds, 2)
    out = {"acc": {}, "lr": {}, "pur": {}, "split_check": {}, "finite": {}}
    for s in SEEDS:
        S1 = {f"S1 ep{e}": np.load(os.path.join(R21, "mlp", ds, f"seed{s}", f"h_ep{e}.npy")) for e in EPOCHS}
        for k in SHOTS:
            ref = rdir("S3", ds, k, s)
            sp = json.load(open(os.path.join(ref, "split.json")))
            labels = np.load(os.path.join(ref, "labels.npy"))
            out["split_check"][f"{k}|{s}"] = {v: json.load(open(os.path.join(rdir(v, ds, k, s), "split.json"))) == sp and
                                              bool(np.array_equal(np.load(os.path.join(rdir(v, ds, k, s), "labels.npy")), labels))
                                              for v in ("S2", "대조")}
            if k == 1:  # S1 trained on the 1-shot T09 split
                out["split_check"][f"{k}|{s}"]["S1 split (1-shot)"] = True
            else:
                sp1 = json.load(open(os.path.join(rdir("S3", ds, 1, s), "split.json")))
                out["split_check"][f"{k}|{s}"]["S1 split (1-shot)"] = sp1 == sp
            groups = {"base": sp["class_list_train"], "valid": sp["class_list_valid"], "test": sp["class_list_test"]}
            by_class = {c: np.where(labels == int(c))[0].tolist() for g in GROUPS for c in groups[g]}
            spaces = {"S0": S0} | S1 | {v: np.load(os.path.join(rdir(v, ds, k, s), "emb_best.npy")) for v in RUNS}
            for n, V in spaces.items():
                out["finite"][f"{n}|{k}|{s}"] = bool(np.isfinite(V).all())
            rng = random.Random(1000 + s)
            for g in GROUPS:
                eps = episodes(rng, groups[g], by_class, k)
                for n, V in spaces.items():
                    out["acc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc(V, eps, k)
                if g == "test":
                    out["lr"][f"{k}|{s}"] = lr_acc(S0, eps, k)
            if k == 5:
                for g in GROUPS:
                    nodes = np.where(np.isin(labels, [int(c) for c in groups[g]]))[0]
                    for n, V in spaces.items():
                        out["pur"][f"{n}|{s}|{g}"] = knn_purity(V, nodes, labels[nodes])  # y aligned with nodes, as analyze_t13.groups_of
            print(ds, s, k, "done", flush=True)
    json.dump(out, open(os.path.join(R21, f"analysis_{ds}.json"), "w"), indent=1)


def t12_table2_test():
    """reports/T12_summary.md 표 2, 그룹 = test: (ds, k) -> (원본 mean, Â²X mean)."""
    lines = open(os.path.join(ROOT, "reports", "T12_summary.md")).read().split("\n")
    i = lines.index("### 그룹 = test")
    hdr = [c.strip() for c in lines[i + 1].strip("|").split("|")]
    out = {}
    for ln in lines[i + 3 : i + 9]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        out[(c[0], int(c[1]))] = (float(c[hdr.index("원본")].split("±")[0]), float(c[hdr.index("Â²X")].split("±")[0]))
    return out


def report():
    A = {ds: json.load(open(os.path.join(R21, f"analysis_{ds}.json"))) for ds in DATASETS}
    mlp = {(ds, s): json.load(open(os.path.join(R21, "mlp", ds, f"seed{s}", "run.json"))) for ds in DATASETS for s in SEEDS}

    def acc(ds, n, k, s, g):
        return A[ds]["acc"][f"{n}|{k}|{s}|{g}"]

    def pur(ds, n, s, g):
        return A[ds]["pur"][f"{n}|{s}|{g}"]

    def m(xs):
        return float(np.mean(xs))

    # ---- gates
    G = {}
    G["Hb"] = {ds: all(all(v.values()) for v in A[ds]["split_check"].values()) for ds in DATASETS}
    # Hc(1) as amended 2026-10-14: final-epoch base training accuracy >= epoch 0 + 40 %p and loss strictly decreasing over
    # the snapshot epochs; reproduction of the first 200-epoch run (run_ep200first.json): |d acc ep200| <= 0.1 %p.
    G["Hc1"] = {}
    for ds in DATASETS:
        for s in SEEDS:
            r, r0 = mlp[(ds, s)], json.load(open(os.path.join(R21, "mlp", ds, f"seed{s}", "run_ep200first.json")))
            a, lo = r["base_train_acc"], r["loss"]
            G["Hc1"][f"{ds}|{s}"] = {"acc_ep0": a[0], "acc_final": a[-1], "gain": a[-1] - a[0],
                                     "loss_snap": [lo[e] for e in EPOCHS], "loss_monotone": all(lo[EPOCHS[i + 1]] < lo[EPOCHS[i]] for i in range(len(EPOCHS) - 1)),
                                     "acc_ep200_first": r0["base_train_acc"][200], "repro_d_ep200": abs(a[200] - r0["base_train_acc"][200])}
    t12 = t12_table2_test()
    G["Hc2"] = {f"{ds}|{n}": {"T21": m([acc(ds, n, 5, s, "test") for s in SEEDS]), "T12": t12[(ds, 5)][0 if n == "S3" else 1]}
                for ds in DATASETS for n in ("S0", "S3")}
    for v in G["Hc2"].values():
        v["diff"] = v["T21"] - v["T12"]
    hd = {}
    for ds, k in CELLS:
        for s in SEEDS:
            r = json.load(open(os.path.join(rdir("S2", ds, k, s), "run.json")))
            hd[f"{ds}|{k}|{s}"] = {"gamma": r["config"]["gamma"], "test_acc_at_best_valid": r["final"].get("test_acc_at_best_valid")}
    G["Hd"] = hd
    vals = [x for ds in DATASETS for key in ("acc", "lr", "pur") for x in A[ds][key].values()]
    G["Hf"] = {"all_metrics_finite": bool(np.isfinite(vals).all()), "n_metrics": len(vals),
               "all_spaces_finite": all(all(A[ds]["finite"].values()) for ds in DATASETS)}
    G["pass"] = {"Hb": all(G["Hb"].values()), "Hc1": all(v["gain"] >= 40 and v["loss_monotone"] for v in G["Hc1"].values()),
                 "Hc1_repro": all(v["repro_d_ep200"] <= 0.1 for v in G["Hc1"].values()),
                 "Hc2": all(abs(v["diff"]) <= 0.5 for v in G["Hc2"].values()),
                 "Hd": all(v["gamma"] == 0.0 and v["test_acc_at_best_valid"] is not None for v in hd.values()),
                 "Hf": G["Hf"]["all_metrics_finite"] and G["Hf"]["all_spaces_finite"]}

    # ---- judgements
    J = {}
    J["J1"] = sum(m([acc(ds, "S1 ep200", k, s, "test") for s in SEEDS]) < m([acc(ds, "S0", k, s, "test") for s in SEEDS]) for ds, k in CELLS)
    J["J2"] = sum(m([acc(ds, "S2", k, s, "test") for s in SEEDS]) < m([acc(ds, "S0", k, s, "test") for s in SEEDS]) for ds, k in CELLS)
    J["J3"] = sum(m([acc(ds, "S1 ep200", k, s, "base") for s in SEEDS]) > m([acc(ds, "S0", k, s, "base") for s in SEEDS]) for ds, k in CELLS)
    j4 = [(k, m([A[ds]["lr"][f"{k}|{s}"] for s in SEEDS]) < m([acc(ds, "S0", k, s, "test") for s in SEEDS])) for ds, k in CELLS]
    J["J4"] = sum(x for _, x in j4)
    J["J4_k1"] = sum(x for k, x in j4 if k == 1)
    J["J5_S1"] = sum(m([pur(ds, "S1 ep200", s, "test") for s in SEEDS]) < m([pur(ds, "S0", s, "test") for s in SEEDS]) for ds in DATASETS)
    J["J5_S2"] = sum(m([pur(ds, "S2", s, "test") for s in SEEDS]) < m([pur(ds, "S0", s, "test") for s in SEEDS]) for ds in DATASETS)
    J["J1'"] = sum(m([acc(ds, "S1 ep1000", k, s, "test") for s in SEEDS]) < m([acc(ds, "S0", k, s, "test") for s in SEEDS]) for ds, k in CELLS)
    J["J3'"] = sum(m([acc(ds, "S1 ep1000", k, s, "base") for s in SEEDS]) > m([acc(ds, "S0", k, s, "base") for s in SEEDS]) for ds, k in CELLS)
    J["J5'_S1"] = sum(m([pur(ds, "S1 ep1000", s, "test") for s in SEEDS]) < m([pur(ds, "S0", s, "test") for s in SEEDS]) for ds in DATASETS)
    J["J6"] = {f"{ds}|{k}": mean_se([acc(ds, "S2", k, s, "test") - acc(ds, "S3", k, s, "test") for s in SEEDS])[0] for ds, k in CELLS}
    P = {"P107": (f"J1 = {J['J1']}/6", J["J1"] >= 5), "P108": (f"J2 = {J['J2']}/6", J["J2"] >= 5),
         "P109": (f"J3 = {J['J3']}/6", J["J3"] >= 5),
         "P110": (f"J4 = {J['J4']}/6, k = 1: {J['J4_k1']}/3", J["J4"] >= 4 and J["J4_k1"] == 3),
         "P111": (f"J5 = {J['J5_S1']}/3, {J['J5_S2']}/3", J["J5_S1"] == 3 and J["J5_S2"] == 3)}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return f"{mean_se(d)[0]} ({sign_agree(d)}/3)"

    L = ["# T21 요약 — 지움의 사다리 (분석 전용)", ""]
    L.append("- S0 = 행 정규화 Â²X, S1 = Â²X → MLP(64) base-CE 은닉층, S2 = TEG γ = 0 GCN 출력, S3 = TEG 원본(T09 `base`), "
             "대조 = T09 `infonce_l10`. seed 0–2.")
    L.append("- probe: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, "
             "그룹 base → valid → test 순으로 한 번 뽑아 모든 표현이 공유). 표현은 미리 정규화하지 않는다(T12와 같음). 정확도 %.")
    L.append("- S2·S3·대조는 probe shot과 같은 shot의 run, S0·S1은 두 k에 같은 표현. 순도는 `analyze_t14.knn_purity`(top-10), "
             "S2·S3·대조는 5-shot run.")
    L.append("")
    L.append("## T1 — 사다리 (seed 평균 ± sd)")
    L.append("| 데이터셋 | k | 표현 | acc_base | acc_valid | acc_test |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for n in LADDER:
            L.append(f"| {ds} | {k} | {n} | " + " | ".join(mean_sd([acc(ds, n, k, s, g) for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T2 — S1 궤적 (seed 평균 ± sd; acc는 k = 5)")
    L.append("| 데이터셋 | 에폭 | base 학습 정확도 | acc_base | acc_test | pur_test |")
    L.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        for e in EPOCHS:
            n = f"S1 ep{e}"
            L.append(f"| {ds} | {e} | {mean_sd([mlp[(ds, s)]['base_train_acc'][e] for s in SEEDS])[0]} | "
                     f"{mean_sd([acc(ds, n, 5, s, 'base') for s in SEEDS])[0]} | {mean_sd([acc(ds, n, 5, s, 'test') for s in SEEDS])[0]} | "
                     f"{mean_sd([pur(ds, n, s, 'test') for s in SEEDS], '{:.4f}')[0]} |")
    L.append("")
    L.append("## T3 — 순도 (top-10 코사인 kNN, seed 평균 ± sd)")
    L.append("| 데이터셋 | 표현 | pur_base | pur_valid | pur_test |")
    L.append("|---|---|---|---|---|")
    for ds in DATASETS:
        for n in LADDER:
            L.append(f"| {ds} | {n} | " + " | ".join(mean_sd([pur(ds, n, s, g) for s in SEEDS], "{:.4f}")[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T4 — head 비교: S0 위 acc_test, 코사인 프로토타입 vs support 로지스틱 회귀")
    L.append("| 데이터셋 | k | 프로토타입 | 로지스틱 회귀 | LR − 프로토타입 (평균 ± SE, 부호 일치/3) |")
    L.append("|---|---|---|---|---|")
    for ds, k in CELLS:
        a = [acc(ds, "S0", k, s, "test") for s in SEEDS]
        b = [A[ds]["lr"][f"{k}|{s}"] for s in SEEDS]
        L.append(f"| {ds} | {k} | {mean_sd(a)[0]} | {mean_sd(b)[0]} | {paired(b, a)} |")
    L.append("")
    L.append("## T5 — seed 단위 짝지은 차 (%p, 평균 ± SE, 부호 일치/3)")
    L.append("| 데이터셋 | k | test: S1 ep200 − S0 | test: S2 − S0 | test: S2 − S3 | base: S1 ep200 − S0 | test: S1 ep1000 − S0 | base: S1 ep1000 − S0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        t = {n: [acc(ds, n, k, s, "test") for s in SEEDS] for n in ("S0", "S1 ep200", "S1 ep1000", "S2", "S3")}
        b = {n: [acc(ds, n, k, s, "base") for s in SEEDS] for n in ("S0", "S1 ep200", "S1 ep1000")}
        L.append(f"| {ds} | {k} | {paired(t['S1 ep200'], t['S0'])} | {paired(t['S2'], t['S0'])} | {paired(t['S2'], t['S3'])} | "
                 f"{paired(b['S1 ep200'], b['S0'])} | {paired(t['S1 ep1000'], t['S0'])} | {paired(b['S1 ep1000'], b['S0'])} |")
    L.append("")
    L.append("## 판정 (값만, seed 평균 기준)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| J1 | acc_test(S1 ep200) < acc_test(S0) | {J['J1']}/6 |")
    L.append(f"| J2 | acc_test(S2) < acc_test(S0) | {J['J2']}/6 |")
    L.append(f"| J3 | acc_base(S1 ep200) > acc_base(S0) | {J['J3']}/6 |")
    L.append(f"| J4 | acc_test^LR(S0) < acc_test(S0); k = 1 | {J['J4']}/6; {J['J4_k1']}/3 |")
    L.append(f"| J5 | pur_test(S1 ep200) < pur_test(S0); pur_test(S2) < pur_test(S0) | {J['J5_S1']}/3; {J['J5_S2']}/3 |")
    j1p, j3p, j5p = J["J1'"], J["J3'"], J["J5'_S1"]
    L.append(f"| J1′ (기록) | acc_test(S1 ep1000) < acc_test(S0) | {j1p}/6 |")
    L.append(f"| J3′ (기록) | acc_base(S1 ep1000) > acc_base(S0) | {j3p}/6 |")
    L.append(f"| J5′ (기록) | pur_test(S1 ep1000) < pur_test(S0) | {j5p}/3 |")
    L.append("| J6 | acc_test(S2) − acc_test(S3) 짝지은 차 | " + "; ".join(f"{c.replace('|', ' ')}s {v}" for c, v in J["J6"].items()) + " |")
    L.append("")
    L.append("## 사전 등록 대조")
    L.append("| 예측 | 조건 | 관측 | 일치 |")
    L.append("|---|---|---|---|")
    cond = {"P107": "J1 ≥ 5/6", "P108": "J2 ≥ 5/6", "P109": "J3 ≥ 5/6", "P110": "J4 ≥ 4/6, k = 1에서 3/3", "P111": "J5 둘 다 3/3"}
    for p, (obs, ok) in P.items():
        L.append(f"| {p} | {cond[p]} | {obs} | {'일치' if ok else '어긋남'} |")
    L.append("")
    open(os.path.join(ROOT, "reports", "T21_summary.md"), "w").write("\n".join(L))

    # ---- figure 1
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["S0", "S1 ep0", "S1 ep200", "S2", "S3", "대조"]
    labels = ["S0\nÂ²X", "S1 ep0", "S1 ep200", "S2\nγ=0", "S3\nTEG", "InfoNCE\nλ=10"]  # figure text in English (no Hangul font)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        mu = [m([acc(ds, n, 5, s, "test") for s in SEEDS]) for n in names]
        sd = [float(np.std([acc(ds, n, 5, s, "test") for s in SEEDS], ddof=1)) for n in names]
        cols = ["#6b7280"] + ["#2563eb"] * 5
        ax.bar(range(len(names)), mu, yerr=sd, color=cols, width=0.7, capsize=3, error_kw={"elinewidth": 1})
        ax.axhline(mu[0], color="#6b7280", lw=1, ls="--")
        for i, v in enumerate(mu):
            ax.text(i, v + sd[i] + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8)
        lo = min(mu) - 8
        ax.set_ylim(lo, max(mu) + 4)
        ax.set_title(f"{ds} (k = 5)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("acc_test (%), mean ± sd over seeds 0–2")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "T21_ladder.png"), dpi=150)
    json.dump({"gates": G, "J": J, "P": {k: list(v) for k, v in P.items()}}, open(os.path.join(R21, "analysis.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(G["pass"]), json.dumps({k: v for k, v in J.items()}, ensure_ascii=False), {k: v[1] for k, v in P.items()})
    print("Hc1", G["Hc1"])
    print("Hc2", {k: round(v["diff"], 3) for k, v in G["Hc2"].items()})


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()
