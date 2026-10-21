"""T28 analysis: pure-X cosine prototype (X0 = row-L2 X, X1 = row-L2 ÂX, D = row-L2 Â²X) on the T22 fixed test episodes.

Episodes: T22 `base` fixed_eval_logits.npz, mode 1 (200 per seed, Random(9000 + seed)), ep_idx order, seed 0-9. TEG / C_all
columns: analyze_t22.fixed_teg on `base` / selected-lambda `infonce_all` npz. T3 (record): T21 probe (analyze_t06b.episodes
+ fewshot_acc, random.Random(1000 + seed), groups base -> valid -> test, seed 0-2, T09 `base` split / labels as in
analyze_t21.run) on X0, X1, S0; S3 copied from results/T21/analysis_<ds>.json. No new RNG, no training.
Usage  python tools/analyze_t28.py run <ds>   -> results/T28/analysis_<ds>.json
       python tools/analyze_t28.py report     -> gates, reports/T28_summary.md, results/T28/gate_checks.txt
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
from analyze_t06b import episodes, fewshot_acc  # noqa: E402  (analyze_t06b chdirs to ROOT)
from analyze_t22 import fixed_teg, mean_sd, paired, run_dir  # noqa: E402

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(10))
PSEEDS = [0, 1, 2]
GROUPS = ["base", "valid", "test"]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
LAM = {("Amazon_clothing", 1): 10, ("Amazon_clothing", 5): 10, ("dblp", 1): 10, ("dblp", 5): 10,
       ("Amazon_electronics", 1): 100, ("Amazon_electronics", 5): 30}  # T22 selected lambda, pool all (reports/T22_summary.md T1)
R22 = os.path.join(ROOT, "results", "T22")
R21 = os.path.join(ROOT, "results", "T21")
R09B = os.path.join(ROOT, "results", "T09", "base")
OUTD = os.path.join(ROOT, "results", "T28")
OUT = os.path.join(ROOT, "reports", "T28_summary.md")
REPS = ["X0", "X1", "D"]
COLS = ["X0", "X1", "D", "TEG", "C_all"]


def npz_path(cfg, ds, k, s):
    return os.path.join(run_dir(cfg, ds, k, s, R22), "fixed_eval_logits.npz")


def call_cfg(ds, k):
    return f"infonce_all_l{LAM[(ds, k)]}"


def load_eps(npz):
    Z = np.load(npz)
    m = np.where(Z["mode"] == 1)[0]
    m = m[np.argsort(Z["ep_idx"][m], kind="stable")]
    return Z["support"][m], Z["query"][m], Z["query_y"][m]


def run(ds):
    torch.set_num_threads(16)
    V = {"X0": fb.diffusion_features_hops(ds, 0), "X1": fb.diffusion_features_hops(ds, 1), "D": fb.diffusion_features(ds)}
    out = {"fixed": {}, "probe": {}, "finite": {n: bool(np.isfinite(v).all()) for n, v in V.items()}, "qy_order": True}
    for k in SHOTS:
        for s in SEEDS:
            sup, qry, qy = load_eps(npz_path("base", ds, k, s))
            out["qy_order"] &= bool((qy == np.repeat(np.arange(5), qry.shape[1] // 5)[None, :]).all())
            for n in REPS:
                out["fixed"][f"{n}|{k}|{s}"] = fb.cosine_probe_acc(V[n], sup, qry, k).tolist()
            out["fixed"][f"TEG|{k}|{s}"] = fixed_teg(npz_path("base", ds, k, s)).tolist()
            out["fixed"][f"C_all|{k}|{s}"] = fixed_teg(npz_path(call_cfg(ds, k), ds, k, s)).tolist()
        print(ds, k, "fixed done", flush=True)
    S0 = fb.diffusion_features_hops(ds, 2)  # T21 S0 call
    P = {"X0": V["X0"], "X1": V["X1"], "S0": S0}
    for s in PSEEDS:
        for k in SHOTS:
            ref = os.path.join(R09B, f"{ds}_5w{k}s", f"seed{s}")
            sp = json.load(open(os.path.join(ref, "split.json")))
            labels = np.load(os.path.join(ref, "labels.npy"))
            groups = {"base": sp["class_list_train"], "valid": sp["class_list_valid"], "test": sp["class_list_test"]}
            by_class = {c: np.where(labels == int(c))[0].tolist() for g in GROUPS for c in groups[g]}
            rng = random.Random(1000 + s)
            for g in GROUPS:
                eps = episodes(rng, groups[g], by_class, k)
                for n, M in P.items():
                    out["probe"][f"{n}|{k}|{s}|{g}"] = fewshot_acc(M, eps, k)
            print(ds, s, k, "probe done", flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(out, open(os.path.join(OUTD, f"analysis_{ds}.json"), "w"))


def t22_table2():
    """reports/T22_summary.md 표 T2: (ds, k) -> {TEG, D, C_all} seed means as printed."""
    lines = open(os.path.join(ROOT, "reports", "T22_summary.md")).read().split("\n")
    i = lines.index("## T2 — 고정 에피소드 (seed 평균 ± sd)")
    hdr = [c.strip() for c in lines[i + 1].strip("|").split("|")]
    out = {}
    for ln in lines[i + 3 : i + 9]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        out[(c[0], int(c[1]))] = {n: float(c[hdr.index(n)].split("±")[0]) for n in ("TEG", "D", "C_all")}
    return out


def report():
    A = {ds: json.load(open(os.path.join(OUTD, f"analysis_{ds}.json"))) for ds in DATASETS}
    T21 = {ds: json.load(open(os.path.join(R21, f"analysis_{ds}.json"))) for ds in DATASETS}
    acc = lambda ds, n, k, s: np.asarray(A[ds]["fixed"][f"{n}|{k}|{s}"])
    smean = lambda ds, n, k: [100 * acc(ds, n, k, s).mean() for s in SEEDS]
    G = []

    # Ga
    n_base = sum(os.path.exists(npz_path("base", ds, k, s)) for ds, k in CELLS for s in SEEDS)
    n_call = sum(os.path.exists(npz_path(call_cfg(ds, k), ds, k, s)) for ds, k in CELLS for s in SEEDS)
    n_t21 = sum(os.path.exists(os.path.join(R21, f"analysis_{ds}.json")) for ds in DATASETS)
    ga = n_base == 60 and n_call == 60 and n_t21 == 3
    G.append(("Ga 선행", f"base npz {n_base}/60, infonce_all(선택 λ) npz {n_call}/60, T21 analysis {n_t21}/3", ga))
    # Gb
    ref = t22_table2()
    gb_rows, gb_max = [], 0.0
    for ds, k in CELLS:
        for n in ("D", "TEG", "C_all"):
            d = abs(np.mean(smean(ds, n, k)) - ref[(ds, k)][n])
            gb_max = max(gb_max, d)
            gb_rows.append((ds, k, n, np.mean(smean(ds, n, k)), ref[(ds, k)][n], d))
    gb = all(r[5] <= 0.005 for r in gb_rows)
    G.append(("Gb 재현", f"18값 \\|Δ\\| 최대 {gb_max:.4f} (기준 ≤ 0.005, T22 표 T2는 소수 둘째 자리)", gb))
    # Gc
    gc_d = [abs(A[ds]["probe"][f"S0|{k}|{s}|{g}"] - T21[ds]["acc"][f"S0|{k}|{s}|{g}"]) for ds in DATASETS for k in SHOTS for s in PSEEDS for g in GROUPS]
    gc = len(gc_d) == 54 and max(gc_d) <= 1e-6
    G.append(("Gc probe 동일", f"S0 {len(gc_d)}값 \\|Δ\\| 최대 {max(gc_d):.2e} (기준 ≤ 1e-6)", gc))
    # Ge
    fin_feat = all(all(A[ds]["finite"].values()) for ds in DATASETS)
    fin_acc = all(np.isfinite(np.asarray(v)).all() for ds in DATASETS for v in A[ds]["fixed"].values()) and \
        all(math.isfinite(v) for ds in DATASETS for v in A[ds]["probe"].values())
    G.append(("Ge 유한", f"특징 유한 {fin_feat}, 지표 유한 {fin_acc}", fin_feat and fin_acc))
    qy = all(A[ds]["qy_order"] for ds in DATASETS)

    L = ["# T28 요약 — 순수 X 코사인 프로토타입 (X0 / X1 / D), T22 고정 test 에피소드", ""]
    L.append("- 에피소드: T22 `base` `fixed_eval_logits.npz` mode 1 (seed당 200, `Random(9000 + seed)`), `ep_idx` 순, seed 0–9 → (ds, k)마다 2,000.")
    L.append("- X0 / X1 / D = 행 L2 정규화 X / ÂX / Â²X (`fusion_baseline.diffusion_features_hops(ds, 0 / 1)`, `diffusion_features`), "
             "`fusion_baseline.cosine_probe_acc`. TEG / C_all = `analyze_t22.fixed_teg` (`base` / 선택 λ `infonce_all`).")
    L.append("- C_all 선택 λ: " + ", ".join(f"{ds} {k}-shot {LAM[(ds, k)]}" for ds, k in CELLS) + ".")
    L.append("- 정확도 %. 짝지은 차: 에피소드 2,000개 평균 ± SE (양수 비율, seed 부호 일치/10). 분명 = 평균 > 0이고 > 2·SE (음수는 부호 반대).")
    L.append(f"- 모든 에피소드의 query_y가 클래스 순(0,…,0,1,…,4): {qy}.")
    L.append("")
    L.append("## T1 — 고정 test 정확도 (seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | " + " | ".join(COLS) + " |")
    L.append("|---|---|" + "---|" * len(COLS))
    M = {}
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd(smean(ds, n, k))[0] for n in COLS) + " |")
        M[(ds, k)] = {n: float(np.mean(smean(ds, n, k))) for n in COLS}
    L.append("")
    PAIRS = [("X0", "D"), ("X0", "TEG"), ("X0", "C_all"), ("X1", "D")]
    PD = {}
    L.append("## T2 — 짝지은 차 (%p)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"{a} − {b}" for a, b in PAIRS) + " |")
    L.append("|---|---|" + "---|" * len(PAIRS))
    for ds, k in CELLS:
        row = []
        for a, b in PAIRS:
            PD[(ds, k, a, b)] = paired([100 * (acc(ds, a, k, s) - acc(ds, b, k, s)) for s in SEEDS])
            row.append(PD[(ds, k, a, b)][0])
        L.append(f"| {ds} | {k} | " + " | ".join(row) + " |")
    L.append("")
    L.append("## T3 (기록) — 표현 수준 probe (T21 probe, seed 0–2, T09 `base` 분할; S3는 T21 전재), seed 평균 ± sd")
    if gc:
        L.append("| 데이터셋 | shot | 표현 | acc_base | acc_valid | acc_test |")
        L.append("|---|---|---|---|---|---|")
        for ds, k in CELLS:
            for n in ("X0", "X1", "S0", "S3"):
                src = T21[ds]["acc"] if n == "S3" else A[ds]["probe"]
                L.append(f"| {ds} | {k} | {n} | " + " | ".join(mean_sd([src[f"{n}|{k}|{s}|{g}"] for s in PSEEDS])[0] for g in GROUPS) + " |")
    else:
        L.append("- Gc 미충족으로 제외.")
    L.append("")

    clear_pos = lambda p: p[1] > 0 and p[1] > 2 * p[2]
    clear_neg = lambda p: p[1] < 0 and -p[1] > 2 * p[2]
    m1 = [c for c in CELLS if M[c]["X0"] > M[c]["D"]]
    m2p = [c for c in CELLS if clear_pos(PD[(*c, "X0", "D")])]
    m2n = [c for c in CELLS if clear_neg(PD[(*c, "X0", "D")])]
    m3 = [c for c in CELLS if M[c]["X0"] > M[c]["TEG"]]
    m4a = [c for c in CELLS if M[c]["X0"] > M[c]["C_all"]]
    m4b = [c for c in CELLS if M[c]["X1"] > M[c]["D"]]
    t3m = lambda ds, n, k, src: np.mean([src[f"{n}|{k}|{s}|test"] for s in PSEEDS])
    m4c = [c for c in CELLS if t3m(c[0], "X0", c[1], A[c[0]]["probe"]) > t3m(c[0], "S3", c[1], T21[c[0]]["acc"])] if gc else None
    fmt = lambda cs: ", ".join(f"{ds} {k}" for ds, k in cs) or "없음"
    L.append("## 판정 (값만, 단위 (ds, k) 6개)")
    L.append("| 판정 | 정의 | 값 | 해당 칸 |")
    L.append("|---|---|---|---|")
    L.append(f"| M1 | acc(X0) > acc(D) (seed 평균) | {len(m1)}/6 | {fmt(m1)} |")
    L.append(f"| M2 | X0 − D 분명한 양수 / 분명한 음수 | {len(m2p)}/6 / {len(m2n)}/6 | 양수: {fmt(m2p)}; 음수: {fmt(m2n)} |")
    L.append(f"| M3 | acc(X0) > acc(TEG) (seed 평균) | {len(m3)}/6 | {fmt(m3)} |")
    L.append(f"| M4 (기록) | acc(X0) > acc(C_all) | {len(m4a)}/6 | {fmt(m4a)} |")
    L.append(f"| M4 (기록) | acc(X1) > acc(D) | {len(m4b)}/6 | {fmt(m4b)} |")
    L.append(f"| M4 (기록) | T3 acc_test(X0) > acc_test(S3) | {'제외' if m4c is None else f'{len(m4c)}/6'} | {'' if m4c is None else fmt(m4c)} |")
    L.append("")
    dblp_neg = [k for k in SHOTS if clear_neg(PD[("dblp", k, "X0", "D")])]
    L.append("## 사전 등록 계산값")
    L.append("| 예측 | 내용 | 계산값 |")
    L.append("|---|---|---|")
    L.append(f"| P144 | M1 = 2/6, 두 칸은 Amazon_electronics 1·5-shot | M1 = {len(m1)}/6 ({fmt(m1)}) |")
    L.append(f"| P145 | dblp 1·5-shot 모두 X0 − D 분명한 음수 | 분명한 음수인 dblp shot: {dblp_neg or '없음'} "
             f"(1-shot {PD[('dblp', 1, 'X0', 'D')][0]}, 5-shot {PD[('dblp', 5, 'X0', 'D')][0]}) |")
    L.append(f"| P146 | M3 ≥ 2/6, Amazon_electronics 2칸 포함 | M3 = {len(m3)}/6 ({fmt(m3)}) |")
    L.append("")
    L.append("## 게이트")
    L.append("| 게이트 | 값 | 충족 |")
    L.append("|---|---|---|")
    for g, v, ok in G:
        L.append(f"| {g} | {v} | {'예' if ok else '아니오'} |")
    L.append("")
    L.append("### Gb 상세 (seed 평균, %)")
    L.append("| 데이터셋 | shot | 열 | T28 | T22 표 T2 | \\|Δ\\| |")
    L.append("|---|---|---|---|---|---|")
    for ds, k, n, a, b, d in gb_rows:
        L.append(f"| {ds} | {k} | {n} | {a:.4f} | {b:.2f} | {d:.4f} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    with open(os.path.join(OUTD, "gate_checks.txt"), "w") as f:
        for g, v, ok in G:
            f.write(f"{g}: {v} -> {'PASS' if ok else 'FAIL'}\n")
    for g, v, ok in G:
        print(g, v, "PASS" if ok else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "report":
        report()
