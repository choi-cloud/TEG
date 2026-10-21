"""T29 analysis -> reports/T29_summary.md, results/T29/gate_checks.txt.

Inputs: results/T29/<ds>/seed<s>/{run.json, per_episode.npz, pretrain.json}, results/T29/_runs.jsonl (driver times),
results/T29/splits.json (Gs). TEG / C_all: analyze_t22.fixed_teg on the T22 `base` / selected-lambda `infonce_all`
fixed_eval_logits.npz (paths and lambda from analyze_t28). Hb compares D, TEG, C_all seed means with reports/T22_summary.md
표 T2 (analyze_t28.t22_table2). Paired differences: analyze_t22.paired over 2,000 episodes. No RNG.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402

from analyze_t22 import fixed_teg, mean_sd, paired  # noqa: E402
from analyze_t28 import CELLS, DATASETS, LAM, SEEDS, call_cfg, npz_path, t22_table2  # noqa: E402

RES = os.path.join(ROOT, "results", "T29")
OUT = os.path.join(ROOT, "reports", "T29_summary.md")
ARMS = ["D", "SGC-P", "SGC-S", "SGC-B", "LR"]
COLS = ARMS + ["TEG", "C_all"]
PAIRS = [("SGC-S", "SGC-B"), ("SGC-P", "SGC-S"), ("D", "SGC-P"), ("D", "SGC-B"), ("C_all", "SGC-P"), ("C_all", "SGC-S"),
         ("TEG", "SGC-B"), ("LR", "D")]
PILOT = {"SGC-P": (92.0, 84.3, 88.3), "SGC-S": (88.6, 84.2, 84.7), "SGC-B": (85.3, 80.9, 81.1)}  # 5-shot, Clothing / dblp / Electronics
GPN_SGC = (62.2, 65.0, 60.8)
PDS = ["Amazon_clothing", "dblp", "Amazon_electronics"]


def load():
    A, P, RJ = {}, {}, {}
    for ds in DATASETS:
        for s in SEEDS:
            d = os.path.join(RES, ds, f"seed{s}")
            if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "per_episode.npz", "pretrain.json")):
                continue
            Z = np.load(os.path.join(d, "per_episode.npz"))
            P[(ds, s)] = json.load(open(os.path.join(d, "pretrain.json")))
            RJ[(ds, s)] = json.load(open(os.path.join(d, "run.json")))
            for k in (1, 5):
                for a in ARMS:
                    A[(ds, k, s, a)] = Z[f"acc_{k}_{a}"]
                A[(ds, k, s, "TEG")] = fixed_teg(npz_path("base", ds, k, s))
                A[(ds, k, s, "C_all")] = fixed_teg(npz_path(call_cfg(ds, k), ds, k, s))
    runs = {}
    rp = os.path.join(RES, "_runs.jsonl")
    if os.path.exists(rp):
        for ln in open(rp):
            r = json.loads(ln)
            runs[(r["dataset"], r["seed"])] = r
    return A, P, RJ, runs


def report():
    A, P, RJ, runs = load()
    sp = json.load(open(os.path.join(RES, "splits.json")))
    G = []
    G.append(("Gs 분할 복원", f"(1)–(5) 실패 목록 {sp['Gs']}", sp["Gs_pass"]))
    hd_bad = [f"{ds} {s}" for (ds, s), p in P.items() if not p["all_finite"]]
    ok_ds = {(ds, s) for (ds, s) in P if f"{ds} {s}" not in hd_bad}
    seeds = {ds: [s for s in SEEDS if (ds, s) in ok_ds] for ds in DATASETS}
    fin = all(np.isfinite(v).all() for v in A.values())
    smean = lambda ds, k, a: [100 * float(A[(ds, k, s, a)].mean()) for s in seeds[ds]]
    ref = t22_table2()
    hb_rows = [(ds, k, a, float(np.mean(smean(ds, k, a))), ref[(ds, k)][a]) for ds, k in CELLS for a in ("D", "TEG", "C_all")]
    hb_max = max(abs(x - y) for *_, x, y in hb_rows)
    full = all(len(seeds[ds]) == 10 for ds in DATASETS)
    G.append(("Hb 재현", f"18값 \\|Δ\\| 최대 {hb_max:.4f} (기준 ≤ 0.005; seed 수 {[len(seeds[ds]) for ds in DATASETS]})", hb_max <= 0.005 and full))
    G.append(("Hd 사전학습", f"라운드 손실 비유한 (ds, seed): {hd_bad or '없음'}; 사전학습 기록 {len(P)}/30", not hd_bad and len(P) == 30))
    G.append(("Hf 유한", f"에피소드 정확도 유한 {fin}", fin))

    L = ["# T29 요약 — 발표된 SGC 기준선의 세 변형, T22 고정 test 에피소드", ""]
    L.append("- 에피소드: T22 `base` `fixed_eval_logits.npz` mode 1 (seed당 200), `ep_idx` 순, seed 0–9 → (ds, k)마다 2,000. 5-way, query 클래스당 5.")
    L.append("- D = 행 L2 Â²X 코사인 프로토타입. V_sgc = Â²(행 L1 X), 사후 정규화 없음. SGC-P = V_sgc 코사인 프로토타입. SGC-S = 무작위 초기화 선형층 → support "
             "`fit_linear`. SGC-B = base 사전학습 선형층(ITER 50 × STEP 50, 클래스당 20) → support `fit_linear`. `fit_linear` = Adam lr 0.2, wd 5e-6, 100 에폭, 전체 배치. "
             "LR = 행 L2 Â²X 위 `LogisticRegression(C=1.0, max_iter=1000)`, support만.")
    L.append("- TEG / C_all = `analyze_t22.fixed_teg` (`base` / 선택 λ `infonce_all`: " + ", ".join(f"{ds} {k} {LAM[(ds, k)]}" for ds, k in CELLS) + ").")
    L.append("- 분할: `results/T29/splits.json` (§2.1 복원). base / valid / test 클래스 수: "
             + ", ".join(f"{ds} {len(sp['splits'][ds]['seeds']['0']['base'])} / {len(sp['splits'][ds]['seeds']['0']['valid'])} / {len(sp['splits'][ds]['test'])}" for ds in DATASETS)
             + " (seed 0–9 같은 크기).")
    L.append("- 정확도 %. 짝지은 차: 에피소드 2,000개 평균 ± SE (양수 비율, seed 부호 일치/10). 분명 = 평균 > 0이고 > 2·SE.")
    L.append("")
    L.append("## T1 — 고정 test 정확도 (seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | " + " | ".join(COLS) + " |")
    L.append("|---|---|" + "---|" * len(COLS))
    M = {}
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd(smean(ds, k, a))[0] for a in COLS) + " |")
        M[(ds, k)] = {a: float(np.mean(smean(ds, k, a))) for a in COLS}
    L.append("")
    PD = {}
    L.append("## T2 — 짝지은 차 (%p)")
    for half in (PAIRS[:4], PAIRS[4:]):
        L.append("| 데이터셋 | shot | " + " | ".join(f"{a} − {b}" for a, b in half) + " |")
        L.append("|---|---|" + "---|" * len(half))
        for ds, k in CELLS:
            row = []
            for a, b in half:
                PD[(ds, k, a, b)] = paired([100 * (A[(ds, k, s, a)] - A[(ds, k, s, b)]) for s in seeds[ds]])
                row.append(PD[(ds, k, a, b)][0])
            L.append(f"| {ds} | {k} | " + " | ".join(row) + " |")
        L.append("")
    L.append("## T3 — 사전학습 기록 (ds, seed)")
    L.append("| 데이터셋 | seed | base 클래스 수 | 마지막 라운드 손실 | 마지막 라운드 정확도 | 라운드 마지막 에폭 손실 최대 | 사전학습 (s) | 특징 (s) | 프로세스 전체 (s, 드라이버) | 시도 | GPU | GPU 최대 (MiB) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for s in SEEDS:
            if (ds, s) not in P:
                L.append(f"| {ds} | {s} | 없음 |||||||||| ")
                continue
            p, rj, r = P[(ds, s)], RJ[(ds, s)], runs.get((ds, s))
            att = r["attempts"][-1] if r else None
            L.append(f"| {ds} | {s} | {p['n_base']} | {p['last_round']['loss']:.4f} | {p['last_round']['acc']:.2f} | {max(p['round_last_epoch_loss']):.4f} | "
                     f"{p['t_pretrain_s']:.1f} | {rj['t_feat_s']:.1f} | {att['sec']:.1f} | {len(r['attempts'])} | {r['gpu']} | {att['gpu_max_mib']} |"
                     if r else f"| {ds} | {s} | {p['n_base']} | {p['last_round']['loss']:.4f} | {p['last_round']['acc']:.2f} | {max(p['round_last_epoch_loss']):.4f} | "
                     f"{p['t_pretrain_s']:.1f} | {rj['t_feat_s']:.1f} | – | – | – | – |")
    L.append("")
    L.append("## T4 (기록) — 5-shot 비교 열")
    L.append("| 데이터셋 | SGC-P (T29) | SGC-P (예비) | SGC-S (T29) | SGC-S (예비) | SGC-B (T29) | SGC-B (예비) | GPN 논문 SGC |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, ds in enumerate(PDS):
        L.append(f"| {ds} | {M[(ds, 5)]['SGC-P']:.2f} | {PILOT['SGC-P'][i]} | {M[(ds, 5)]['SGC-S']:.2f} | {PILOT['SGC-S'][i]} | "
                 f"{M[(ds, 5)]['SGC-B']:.2f} | {PILOT['SGC-B'][i]} | {GPN_SGC[i]} |")
    L.append("- 예비 실험(`instructions/ref/pilot_sgc_vs_d.py`)은 이 표와 분할(`random.seed(seed)` valid 추출, `*_test.mat` test 클래스)과 "
             "query 수(클래스당 20 vs 5), 에피소드(`random.Random(1000 + seed)`, 200개)가 다르다. GPN 논문 값은 논문 보고값이다.")
    L.append("")

    clear_pos = lambda p: p[1] > 0 and p[1] > 2 * p[2]
    fmt = lambda cs: ", ".join(f"{ds} {k}" for ds, k in cs) or "없음"
    n1 = [c for c in CELLS if clear_pos(PD[(*c, "SGC-S", "SGC-B")])]
    n2 = [c for c in CELLS if M[c]["SGC-S"] < M[c]["SGC-P"]]
    n3 = [c for c in CELLS if M[c]["SGC-B"] < M[c]["D"]]
    n4 = [c for c in CELLS if M[c]["C_all"] > M[c]["SGC-P"]]
    n5 = [c for c in CELLS if M[c]["TEG"] > M[c]["SGC-B"]]
    n6a = [c for c in CELLS if clear_pos(PD[(*c, "LR", "D")])]
    n6b = max(abs(M[c]["D"] - M[c]["SGC-P"]) for c in CELLS)
    n6b_c = max(CELLS, key=lambda c: abs(M[c]["D"] - M[c]["SGC-P"]))
    L.append("## 판정 (값만, 단위 (ds, k) 6개)")
    L.append("| 판정 | 정의 | 값 | 해당 칸 |")
    L.append("|---|---|---|---|")
    L.append(f"| N1 | SGC-S − SGC-B 분명한 양수 | {len(n1)}/6 | {fmt(n1)} |")
    L.append(f"| N2 | acc(SGC-S) < acc(SGC-P) (seed 평균) | {len(n2)}/6 | {fmt(n2)} |")
    L.append(f"| N3 | acc(SGC-B) < acc(D) (seed 평균) | {len(n3)}/6 | {fmt(n3)} |")
    L.append(f"| N4 | acc(C_all) > acc(SGC-P) (seed 평균) | {len(n4)}/6 | {fmt(n4)} |")
    L.append(f"| N5 | acc(TEG) > acc(SGC-B) (seed 평균) | {len(n5)}/6 | {fmt(n5)} |")
    L.append(f"| N6 (기록) | LR − D 분명한 양수 | {len(n6a)}/6 | {fmt(n6a)} |")
    L.append(f"| N6 (기록) | \\|D − SGC-P\\| 최대 (seed 평균, %p) | {n6b:.2f} | {fmt([n6b_c])} |")
    L.append("")
    L.append("## 사전 등록 계산값")
    L.append("| 예측 | 내용 | 계산값 |")
    L.append("|---|---|---|")
    L.append(f"| P147 | N1 = 6/6 | N1 = {len(n1)}/6 |")
    L.append(f"| P148 | N2 ≥ 5/6 | N2 = {len(n2)}/6 |")
    L.append(f"| P149 | N3 = 6/6 | N3 = {len(n3)}/6 |")
    L.append(f"| P150 | N4 ≥ 4/6 | N4 = {len(n4)}/6 |")
    L.append(f"| P151 | N5 = 6/6 | N5 = {len(n5)}/6 |")
    L.append("")
    L.append("## 게이트")
    L.append("| 게이트 | 값 | 충족 |")
    L.append("|---|---|---|")
    for g, v, ok in G:
        L.append(f"| {g} | {v} | {'예' if ok else '아니오'} |")
    L.append("")
    L.append("### Hb 상세 (seed 평균, %)")
    L.append("| 데이터셋 | shot | 열 | T29 | T22 표 T2 | \\|Δ\\| |")
    L.append("|---|---|---|---|---|---|")
    for ds, k, a, x, y in hb_rows:
        L.append(f"| {ds} | {k} | {a} | {x:.4f} | {y:.2f} | {abs(x - y):.4f} |")
    L.append("")
    qy = all(rj["shots"][str(k)]["qy_class_order"] for rj in RJ.values() for k in (1, 5))
    L.append(f"- 모든 에피소드의 query_y가 클래스 순: {qy}. 장치: {sorted({rj['device'] for rj in RJ.values()})}.")
    open(OUT, "w").write("\n".join(L) + "\n")
    with open(os.path.join(RES, "gate_checks.txt"), "w") as f:
        for g, v, ok in G:
            f.write(f"{g}: {v} -> {'PASS' if ok else 'FAIL'}\n")
    for g, v, ok in G:
        print(g, v, "PASS" if ok else "FAIL")


if __name__ == "__main__":
    report()
