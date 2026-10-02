"""T05 analysis: results/T05/**/{run.json, edges_test.npz} (+ T04 for regression) -> reports/T05_summary.md.

All tables use, per run, only the test edges of off's best_valid_epoch (same rule as T04 table 3).
Distances are sign-flipped so that larger = same class.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R05 = os.path.join(ROOT, "results", "T05")
R04 = os.path.join(ROOT, "results", "T04")
OUT = os.path.join(ROOT, "reports", "T05_summary.md")

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2, 3, 4]
KEYS = ["raw", "center", "str"]
G1_5SHOT_TOL = 0.0024  # T03b redefinition (max |diff| between original repeat runs)
G2_TOL = 0.002


def auc(y, score):
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None


def mean_sd(xs, fmt="{:.4f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "null", None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def sign_agree(xs):
    m = sum(xs) / len(xs)
    sg = (m > 0) - (m < 0)
    return sum(1 for x in xs if sg != 0 and ((x > 0) - (x < 0)) == sg)


def oof_auc(X, y, groups):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    prob = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def load():
    runs, missing = {}, []
    for ds in DATASETS:
        for k in SHOTS:
            for s in SEEDS:
                d = os.path.join(R05, f"{ds}_5w{k}s", f"seed{s}")
                if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "edges_test.npz")):
                    missing.append((ds, k, s))
                    continue
                run = json.load(open(os.path.join(d, "run.json")))
                edges = dict(np.load(os.path.join(d, "edges_test.npz")))
                ref = json.load(open(os.path.join(R04, f"{ds}_5w{k}s", f"seed{s}", "run.json")))
                runs[(ds, k, s)] = (run, edges, ref)
    return runs, missing


def per_run(run, edges):
    e = run["final"]["best_epoch_valid"]
    m = edges["epoch"] == e
    if not m.any():
        return None
    E = {k: v[m] for k, v in edges.items()}
    y = E["same"].astype(int)
    g = E["ep_idx"]
    feats = {"-d0": -E["d0"], "-d_final": -E["d_final"]} | {f"phat_{k}": E[f"phat_{k}"] for k in KEYS}
    out = {"epoch": e, "n": len(y)}
    out["A"] = {f: auc(y, v) for f, v in feats.items()}
    for base in ("-d0", "-d_final"):
        b = oof_auc(feats[base][:, None], y, g)
        out[f"B_base_{base}"] = b
        for key in KEYS:
            out[f"B_{base}_{key}"] = oof_auc(np.stack([feats[base], feats[f"phat_{key}"]], 1), y, g) - b
    q25, q75 = np.percentile(E["d0"], [25, 75])
    sub = ((y == 1) & (E["d0"] >= q75)) | ((y == 0) & (E["d0"] <= q25))
    out["C_n1"], out["C_n0"] = int((y[sub] == 1).sum()), int((y[sub] == 0).sum())
    out["C"] = {f: auc(y[sub], v[sub]) for f, v in feats.items()}
    out["D"] = {k: (float(E[f"top1_{k}"].mean()), float((E[f"top1_{k}"] - E[f"top10_{k}"]).mean())) for k in KEYS}
    return out


def main():
    runs, missing = load()
    stats = {key: per_run(run, edges) for key, (run, edges, _) in runs.items()}
    no_edges = [key for key, v in stats.items() if v is None]
    by_cell = defaultdict(list)
    for (ds, k, s), v in stats.items():
        if v is not None:
            by_cell[(ds, k)].append(v)
    cells = [(ds, k) for ds in DATASETS for k in SHOTS]

    L = ["# T05 요약 — 메모리 보완성 진단", ""]
    L.append(f"- run 수: **{len(runs)} / 30** (`run.json`·`edges_test.npz` 존재). 누락: {missing or '없음'}. "
             f"best_valid_epoch = 0 으로 간선이 없는 run: {no_edges or '없음'}.")
    L.append("- 모든 표: run마다 off의 `best_valid_epoch` test 간선만 사용. 값 = (데이터셋, shot)별 seed 평균 ± sd. 거리는 부호를 뒤집어 사용.")
    L.append("")

    # Table A
    feats = ["-d0", "-d_final"] + [f"phat_{k}" for k in KEYS]
    L.append("## 표 A — 단일 점수 AUC")
    L.append("| 데이터셋 | shot | n | " + " | ".join(f"`{f}`" for f in feats) + " |")
    L.append("|---|---|---|" + "---|" * len(feats))
    p39 = {}
    for c in cells:
        vs = by_cell[c]
        row = [mean_sd([v["A"][f] for v in vs]) for f in feats]
        p39[c] = (row[1][1], row[0][1])
        L.append(f"| {c[0]} | {c[1]} | {len(vs)} | " + " | ".join(r[0] for r in row) + " |")
    L.append("")

    # Table B
    L.append("## 표 B — 보완성 (로지스틱 회귀, 표준화, sklearn 기본 L2, `ep_idx` 5겹 GroupKFold, out-of-fold AUC)")
    L.append("셀 = ΔAUC (비교 − 기준선) seed 평균 ± sd, 부호 일치 수/seed 수.")
    L.append("")
    p37, p38 = {}, {}
    for base in ("-d0", "-d_final"):
        L.append(f"### 기준선 `[{base}]`")
        L.append(f"| 데이터셋 | shot | 기준선 AUC | " + " | ".join(f"+`phat_{k}`" for k in KEYS) + " |")
        L.append("|---|---|---|" + "---|" * len(KEYS))
        for c in cells:
            vs = by_cell[c]
            row = [mean_sd([v[f"B_base_{base}"] for v in vs])[0]]
            for key in KEYS:
                xs = [v[f"B_{base}_{key}"] for v in vs]
                txt, m = mean_sd(xs, "{:+.4f}")
                row.append(f"{txt}, {sign_agree(xs)}/{len(xs)}")
                if base == "-d0":
                    p38.setdefault(c, {})[key] = m
                    if key == "raw":
                        p37[c] = m
            L.append(f"| {c[0]} | {c[1]} | " + " | ".join(row) + " |")
        L.append("")

    # Table C
    L.append("## 표 C — 거리가 헷갈리는 쌍 (same=1 & d0 ≥ 75백분위) ∪ (same=0 & d0 ≤ 25백분위)")
    L.append("| 데이터셋 | shot | same=1 개수 | same=0 개수 | " + " | ".join(f"`{f}`" for f in feats) + " |")
    L.append("|---|---|---|---|" + "---|" * len(feats))
    for c in cells:
        vs = by_cell[c]
        row = [mean_sd([v["C_n1"] for v in vs], "{:.1f}")[0], mean_sd([v["C_n0"] for v in vs], "{:.1f}")[0]]
        row += [mean_sd([v["C"][f] for v in vs])[0] for f in feats]
        L.append(f"| {c[0]} | {c[1]} | " + " | ".join(row) + " |")
    L.append("")

    # Table D
    L.append("## 표 D — 키별 조회 유사도 분포")
    L.append("| 데이터셋 | shot | " + " | ".join(f"`top1_{k}` 평균 | `top1−top10_{k}` 평균" for k in KEYS) + " |")
    L.append("|---|---|" + "---|---|" * len(KEYS))
    for c in cells:
        vs = by_cell[c]
        row = []
        for key in KEYS:
            row.append(mean_sd([v["D"][key][0] for v in vs], "{:.6f}")[0])
            row.append(mean_sd([v["D"][key][1] for v in vs], "{:.6f}")[0])
        L.append(f"| {c[0]} | {c[1]} | " + " | ".join(row) + " |")
    L.append("")

    # Table E + gates
    L.append("## 표 E — 회귀 확인 (T04 같은 seed 대비)")
    L.append("| 데이터셋 | shot | seed | off 정확도 33개 일치 수 | `test_acc_at_best_valid` 일치 | valid/test 최대 \\|차\\| | train 최대 \\|차\\| | max \\|`auc_phat_pooled_raw` − T04 `auc_phat_pooled`\\| | `auc_phat_pooled_raw` = 자기 `auc_phat_pooled` |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    g1_fail, g2_max, self_eq_all = [], 0.0, True
    for (ds, k, s), (run, _, ref) in sorted(runs.items(), key=lambda x: (DATASETS.index(x[0][0]), x[0][1], x[0][2])):
        same_n = sum(a[m] == b[m] for a, b in zip(run["epochs"], ref["epochs"]) for m in ("train_acc", "valid_acc", "test_acc"))
        vt = max(abs(a[m] - b[m]) for a, b in zip(run["epochs"], ref["epochs"]) for m in ("valid_acc", "test_acc"))
        tr = max(abs(a["train_acc"] - b["train_acc"]) for a, b in zip(run["epochs"], ref["epochs"]))
        fin = run["final"]["test_acc_at_best_valid"] == ref["final"]["test_acc_at_best_valid"]
        d2 = max(abs(a["auc_phat_pooled_raw"] - b["auc_phat_pooled"]) for a, b in zip(run["epochs"][1:], ref["epochs"][1:]))
        self_eq = all(a["auc_phat_pooled_raw"] == a["auc_phat_pooled"] for a in run["epochs"][1:])
        self_eq_all &= self_eq
        g2_max = max(g2_max, d2)
        if k == 1 and not (same_n == 33 and fin):
            g1_fail.append((ds, k, s))
        if k == 5 and not (vt <= G1_5SHOT_TOL + 1e-12):
            g1_fail.append((ds, k, s))
        L.append(f"| {ds} | {k} | {s} | {same_n}/33 | {'예' if fin else '아니오'} | {vt:.4f} | {tr:.4f} | {d2:.6f} | {'예' if self_eq else '아니오'} |")
    L.append("")
    L.append("## 게이트")
    L.append("| 게이트 | 조건 | 관측 | 결과 |")
    L.append("|---|---|---|---|")
    L.append(f"| G1 | 1-shot 33개·`test_acc_at_best_valid` 완전 일치, 5-shot valid/test 최대 \\|차\\| ≤ {G1_5SHOT_TOL} | 불충족 run: {g1_fail or '없음'} | {'통과' if not g1_fail else '실패'} |")
    L.append(f"| G2 | 전 에폭 \\|`auc_phat_pooled_raw` − T04\\| ≤ {G2_TOL} | 최대 {g2_max:.6f} | {'통과' if g2_max <= G2_TOL else '실패'} |")
    L.append(f"| G3 | 30 run 파일 존재, 표 A–E 생성 | {len(runs)}/30, 표 생성 | {'통과' if len(runs) == 30 else '실패'} |")
    L.append("")

    # P37-P39
    n37 = sum(1 for v in p37.values() if v < 0.01)
    n38 = sum(1 for c in p38 if any(v >= 0.01 for v in p38[c].values()))
    n39 = sum(1 for a, b in p39.values() if a > b)
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P37 | 기준선 1 `+phat_raw` ΔAUC < 0.01, 6개 모두 | {n37}/6 (값: " + ", ".join(f"{p37[c]:+.4f}" for c in cells) + f") | {'일치' if n37 == 6 else '어긋남'} |")
    L.append(f"| P38 | 기준선 1 세 키 중 하나라도 ΔAUC ≥ 0.01인 (데이터셋, shot) ≤ 2개 | {n38}개 | {'일치' if n38 <= 2 else '어긋남'} |")
    L.append(f"| P39 | AUC(`−d_final`) > AUC(`−d0`), 6개 모두 | {n39}/6 | {'일치' if n39 == 6 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}; runs={len(runs)} G1_fail={g1_fail} G2_max={g2_max:.6f} self_eq={self_eq_all}")


if __name__ == "__main__":
    main()
