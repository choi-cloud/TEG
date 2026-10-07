"""T12 analysis (train accuracy vs unseen-class separability; analysis only) -> reports/T12_summary.md, results/T12_gate_checks.txt.

Table 1: T10r runs (seed 0-9), base / kl_h2 / infonce with the selected lambda read from reports/T10r_summary.md table 1;
         train = run.json epochs[ckpt_epoch]["train_acc"], valid = final best_acc_valid, test = final test_acc_at_best_valid.
Table 2: T09 emb_best.npy (seed 0-2), base / kl_h2 / infonce with the selected lambda read from reports/T09_summary.md table 2,
         plus row-normalized A_hat^1 X and A_hat^2 X; probe = tools/analyze_t06b.py `episodes` + `fewshot_acc` (T06b table B1),
         episodes from random.Random(1000 + seed), drawn once per group (base -> valid -> test) and shared by all spaces.
"""
import json
import math
import os
import random
import re
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import fusion_baseline as fb  # noqa: E402
from analyze_t06b import episodes, fewshot_acc  # noqa: E402  (T06b table B1 probe)

ROOT = fb.ROOT
R10 = os.path.join(ROOT, "results", "T10r")
R09 = os.path.join(ROOT, "results", "T09")
OUT = os.path.join(ROOT, "reports", "T12_summary.md")
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
GROUPS = ["base", "valid", "test"]
VAR = ["원본", "보존", "대조"]


def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def seed_paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
    agree = int(np.sum(np.sign(d) == np.sign(m))) if m != 0 else 0
    return f"{m:+.2f} ± {se:.2f} ({agree}/{n})", m, se


def table_rows(path, header):
    lines = open(path).read().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith(header))
    out = []
    for l in lines[i + 3:]:
        if not l.startswith("|"):
            break
        out.append([c.strip() for c in l.strip("|").split("|")])
    return out


def main():
    # ---- selected lambda from the experiment summaries
    t10_rows = {(r[0], int(r[1])): r for r in table_rows(os.path.join(ROOT, "reports", "T10r_summary.md"), "## 표 1")}
    sel10 = {}
    for c, r in t10_rows.items():
        kl, inf = [x.replace("‡", "").strip() for x in r[3].split("/")]
        sel10[c] = {"원본": "base", "보존": f"kl_h2_l{kl}", "대조": f"infonce_l{inf}"}
    sel09 = {}
    for r in table_rows(os.path.join(ROOT, "reports", "T09_summary.md"), "## 표 2"):
        if r[2] in ("kl_h2", "infonce"):
            sel09.setdefault((r[0], int(r[1])), {"원본": "base"})["보존" if r[2] == "kl_h2" else "대조"] = f"{r[2]}_l{r[3]}"

    # ---- table 1 (T10r, seed 0-9)
    T1 = {}
    for ds, k in CELLS:
        for v in VAR:
            for s in range(10):
                run = json.load(open(os.path.join(R10, sel10[(ds, k)][v], f"{ds}_5w{k}s", f"seed{s}", "run.json")))
                e = run["ckpt_epoch"]
                T1[(ds, k, v, s)] = {"train": 100 * run["epochs"][e]["train_acc"], "valid": 100 * run["final"]["best_acc_valid"],
                                     "test": 100 * run["final"]["test_acc_at_best_valid"], "valid_at_ckpt": 100 * run["epochs"][e]["valid_acc"]}

    L = ["# T12 요약 — 학습 정확도 vs 안 본 클래스 구분력 (분석 전용)", ""]
    L.append("- 원본 = `base`(λ = 0), 보존 = `kl_h2`(선택 λ), 대조 = `infonce`(선택 λ). 표 1 선택 λ는 `reports/T10r_summary.md` 표 1, 표 2 선택 λ는 `reports/T09_summary.md` 표 2에서 읽었다.")
    L.append("- train = `run.json` `epochs[ckpt_epoch].train_acc`(그 에폭 학습 에피소드 50개, optimizer step 전 forward 출력, 학습 모드), valid = `best_acc_valid`, test = `test_acc_at_best_valid`. 정확도 %.")
    L.append("")
    L.append("## 표 1 — 정확도 세 가지 (T10r, seed 0–9, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 선택 λ (보존 / 대조) | " + " | ".join(f"{v} {m}" for v in VAR for m in ("train", "valid", "test")) + " | " +
             " | ".join(f"{v} train − test" for v in VAR) + " |")
    L.append("|---|---|---|" + "---|" * 12)
    for ds, k in CELLS:
        cols = []
        for v in VAR:
            for m in ("train", "valid", "test"):
                cols.append(mean_sd([T1[(ds, k, v, s)][m] for s in range(10)])[0])
        for v in VAR:
            cols.append(mean_sd([T1[(ds, k, v, s)]["train"] - T1[(ds, k, v, s)]["test"] for s in range(10)], "{:+.2f}")[0])
        lam = f"{sel10[(ds, k)]['보존'].rsplit('_l', 1)[1]} / {sel10[(ds, k)]['대조'].rsplit('_l', 1)[1]}"
        L.append(f"| {ds} | {k} | {lam} | " + " | ".join(cols) + " |")
    L.append("")
    L.append("### 표 1 — seed 단위 짝지은 차 (%p, 평균 ± SE, 부호 일치 수/10)")
    L.append("| 데이터셋 | shot | 보존 − 원본 train | 보존 − 원본 valid | 보존 − 원본 test | 대조 − 원본 train | 대조 − 원본 valid | 대조 − 원본 test |")
    L.append("|---|---|---|---|---|---|---|---|")
    J = {f"R_tr {v}": 0 for v in ("보존", "대조")} | {f"R_te {v}": 0 for v in ("보존", "대조")} | {f"R_gap {v}": 0 for v in ("보존", "대조")}
    for ds, k in CELLS:
        cols = []
        for v in ("보존", "대조"):
            for m in ("train", "valid", "test"):
                txt, mm, se = seed_paired([T1[(ds, k, v, s)][m] for s in range(10)], [T1[(ds, k, "원본", s)][m] for s in range(10)])
                cols.append(txt)
                if m == "train":
                    J[f"R_tr {v}"] += abs(mm) < 1.0
                if m == "test":
                    J[f"R_te {v}"] += mm > 0 and mm > 2 * se
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")

    # ---- table 2 (T09 emb_best.npy, seed 0-2)
    T2 = {}
    split_same = []
    for ds in DATASETS:
        H1 = fb.diffusion_features_hops(ds, 1)
        H2 = fb.diffusion_features(ds)
        for k in SHOTS:
            for s in range(3):
                dirs = {v: os.path.join(R09, sel09[(ds, k)][v], f"{ds}_5w{k}s", f"seed{s}") for v in VAR}
                splits = {v: json.load(open(os.path.join(d, "split.json"))) for v, d in dirs.items()}
                labels = np.load(os.path.join(dirs["원본"], "labels.npy"))
                split_same.append(all(splits[v] == splits["원본"] for v in VAR) and
                                  all(np.array_equal(np.load(os.path.join(dirs[v], "labels.npy")), labels) for v in VAR))
                sp = splits["원본"]
                groups = {"base": sp["class_list_train"], "valid": sp["class_list_valid"], "test": sp["class_list_test"]}
                by_class = {c: np.where(labels == int(c))[0].tolist() for g in GROUPS for c in groups[g]}
                spaces = {v: np.load(os.path.join(dirs[v], "emb_best.npy")) for v in VAR} | {"Â¹X": H1, "Â²X": H2}
                rng = random.Random(1000 + s)
                for g in GROUPS:
                    eps = episodes(rng, groups[g], by_class, k)
                    for name, V in spaces.items():
                        T2[(ds, k, s, name, g)] = fewshot_acc(V, eps, k)
            print(ds, "done", flush=True)
        del H1, H2

    names = VAR + ["Â¹X", "Â²X"]
    L.append("## 표 2 — 클래스 그룹별 표현 구분력 (T09 `emb_best.npy`, seed 0–2, 코사인 프로토타입 probe 500 에피소드, %)")
    for g in GROUPS:
        L.append(f"### 그룹 = {g}")
        L.append("| 데이터셋 | shot | 선택 λ (보존 / 대조) | " + " | ".join(names) + " |")
        L.append("|---|---|---|" + "---|" * len(names))
        for ds, k in CELLS:
            lam = f"{sel09[(ds, k)]['보존'].rsplit('_l', 1)[1]} / {sel09[(ds, k)]['대조'].rsplit('_l', 1)[1]}"
            L.append(f"| {ds} | {k} | {lam} | " + " | ".join(mean_sd([T2[(ds, k, s, n, g)] for s in range(3)])[0] for n in names) + " |")
        L.append("")
    L.append("### 표 2 — Δ = 보조 손실 모델 − 원본 (%p, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 보존 Δ_base | 보존 Δ_valid | 보존 Δ_test | 대조 Δ_base | 대조 Δ_valid | 대조 Δ_test |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        cols = []
        for v in ("보존", "대조"):
            dm = {}
            for g in GROUPS:
                txt, dm[g] = mean_sd([T2[(ds, k, s, v, g)] - T2[(ds, k, s, "원본", g)] for s in range(3)], "{:+.2f}")
                cols.append(txt)
            J[f"R_gap {v}"] += dm["test"] > dm["base"]
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")

    L.append("## 판정 (값만)")
    L.append("| 판정 | 정의 | 보존 | 대조 |")
    L.append("|---|---|---|---|")
    L.append(f"| R_tr | 표 1 \\|train 짝지은 차\\| < 1.0%p 인 (데이터셋, shot) 수 | {J['R_tr 보존']}/6 | {J['R_tr 대조']}/6 |")
    L.append(f"| R_te | 표 1 test 짝지은 차 평균 > 0 이고 > 2·SE 인 (데이터셋, shot) 수 | {J['R_te 보존']}/6 | {J['R_te 대조']}/6 |")
    L.append(f"| R_gap | 표 2 Δ_test > Δ_base 인 (데이터셋, shot) 수 | {J['R_gap 보존']}/6 | {J['R_gap 대조']}/6 |")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    ok71 = J["R_te 보존"] == 6 and J["R_te 대조"] == 6
    ok72 = J["R_tr 보존"] >= 4 and J["R_tr 대조"] >= 4
    ok73 = J["R_gap 보존"] >= 5 and J["R_gap 대조"] >= 5
    L.append(f"| P71 | R_te = 6/6 (보존, 대조 모두) | 보존 {J['R_te 보존']}/6, 대조 {J['R_te 대조']}/6 | {'일치' if ok71 else '어긋남'} |")
    L.append(f"| P72 | R_tr ≥ 4/6 (보존, 대조 모두) | 보존 {J['R_tr 보존']}/6, 대조 {J['R_tr 대조']}/6 | {'일치' if ok72 else '어긋남'} |")
    L.append(f"| P73 | R_gap ≥ 5/6 (보존, 대조 모두) | 보존 {J['R_gap 보존']}/6, 대조 {J['R_gap 대조']}/6 | {'일치' if ok73 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")

    # ---- gate G2: table 1 test columns vs reports/T10r_summary.md table 1
    g2 = []
    for (ds, k), r in t10_rows.items():
        for v, col in (("원본", 4), ("보존", 6), ("대조", 8)):
            mine = mean_sd([T1[(ds, k, v, s)]["test"] for s in range(10)])[0]
            g2.append(((ds, k, v), mine, r[col], mine == r[col]))
    valid_eq = sum(abs(x["valid"] - x["valid_at_ckpt"]) < 1e-9 for x in T1.values())
    G = [f"G2 table 1 test == T10r_summary table 1: {sum(x[3] for x in g2)}/{len(g2)}; mismatches {[x for x in g2 if not x[3]] or 'none'}",
         f"valid (best_acc_valid) == epochs[ckpt_epoch].valid_acc: {valid_eq}/{len(T1)}",
         f"table 2 splits/labels identical across 원본/보존/대조 per (ds, shot, seed): {sum(split_same)}/{len(split_same)}",
         f"sel10 {sel10}", f"sel09 {sel09}", f"J {J}"]
    open(os.path.join(ROOT, "results", "T12_gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main()
