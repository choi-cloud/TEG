"""T07 analysis (E1, Preserve only) -> reports/T07_summary.md.

Inputs: results/T07/<cfg>/<ds>_5w<k>s/seed<s>/{run.json, fixed_eval_logits.npz, emb_best.npy}.
All comparisons on the fixed episodes (test 200 / valid 100 per seed), shared by the 7 configs of a (dataset, shot, seed).
View D, combination F and probes via tools/fusion_baseline.py. Labels are used for analysis only. No RNG.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_baseline as fb  # noqa: E402

ROOT = fb.ROOT
R07 = os.path.join(ROOT, "results", "T07")
OUT = os.path.join(ROOT, "reports", "T07_summary.md")

CONFIGS = ["base", "l0.1_nb", "l1_nb", "l10_nb", "l0.1_all", "l1_all", "l10_all"]
LAM = {"base": 0.0, "l0.1_nb": 0.1, "l1_nb": 1.0, "l10_nb": 10.0, "l0.1_all": 0.1, "l1_all": 1.0, "l10_all": 10.0}
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]


def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def paired(diff_by_seed):
    d = np.concatenate(diff_by_seed)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    sg = (m > 0) - (m < 0)
    agree = sum(1 for x in diff_by_seed if sg != 0 and ((float(x.mean()) > 0) - (float(x.mean()) < 0)) == sg)
    return f"{m:+.2f} ± {se:.2f} ({np.mean(d > 0):.2f}, {agree}/{len(diff_by_seed)})", m, se


def run_dir(cfg, ds, k, s):
    return os.path.join(R07, cfg, f"{ds}_5w{k}s", f"seed{s}")


def fixed_test(npz):
    Z = dict(np.load(npz))
    m = np.where(Z["mode"] == 1)[0]
    m = m[np.argsort(Z["ep_idx"][m], kind="stable")]
    return Z["support"][m], Z["query"][m], Z["query_y"][m].astype(int), Z["logits"][m]


def main():
    R = {}
    missing = []
    for ds in DATASETS:
        H2 = fb.diffusion_features(ds)
        H1 = fb.diffusion_features_hops(ds, 1)
        for k in SHOTS:
            for s in SEEDS:
                for cfg in CONFIGS:
                    d = run_dir(cfg, ds, k, s)
                    if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "fixed_eval_logits.npz", "emb_best.npy")):
                        missing.append((cfg, ds, k, s))
                        continue
                    run = json.load(open(os.path.join(d, "run.json")))
                    npz = os.path.join(d, "fixed_eval_logits.npz")
                    sup, qry, y, prob = fixed_test(npz)
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    emb = np.load(os.path.join(d, "emb_best.npy"))
                    R[(cfg, ds, k, s)] = {
                        "bav": run["final"]["best_acc_valid"], "tab": run["final"]["test_acc_at_best_valid"],
                        "teg": (prob.argmax(-1) == y).mean(-1), "F": ev["acc"], "D": ev["acc_D"], "F_TL": ev["T_L"], "F_s": ev["s"],
                        "F_edge": ev["iT"] in (0, len(fb.TL_GRID) - 1) or ev["iS"] in (0, len(fb.S_GRID) - 1),
                        "pres_last": run["epochs"][-1].get("pres_loss_mean"),
                        "probe_emb": fb.cosine_probe_acc(emb, sup, qry, k),
                        "probe_h1": fb.cosine_probe_acc(H1, sup, qry, k) if cfg == "base" else None,
                    }
            print(ds, k, "done", flush=True)
        del H1, H2

    n_runs = len(R)
    complete = lambda ds, k: all((cfg, ds, k, s) in R for cfg in CONFIGS for s in SEEDS)
    L = ["# T07 요약 — E1: 보존 손실(Preserve)만 추가한 TEG, 고정 에피소드 비교", ""]
    L.append(f"- run 수: **{n_runs} / 126** (7 설정 × 3 데이터셋 × 5-way {{1,5}}-shot × seed 0–2). 누락: {missing or '없음'}.")
    L.append("- 정확도 %, (데이터셋, shot)별 seed 3개 평균 ± sd. 짝지은 차: 고정 test 에피소드 600개(200 × seed 3) 평균 ± SE (양수 비율, seed 부호 일치 수/3), %p.")
    L.append("- TEG+P(cfg) = cfg 모델(선택 체크포인트)의 고정 test 정확도. D = 관점 D 단독. F(cfg) = cfg 덤프 + 관점 D, α = 0.5, 온도는 같은 파일의 고정 valid로 보정(`fusion_baseline.evaluate_npz`, log).")
    L.append("")

    # ---- table 1
    L.append("## 표 1 — 전체 격자")
    L.append("| 데이터셋 | shot | 설정 | 원본 best_acc_valid | 원본 test_acc_at_best_valid | 고정 test (TEG+P) | F(cfg) 고정 test | 마지막 에폭 pres_loss_mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    sel = {}
    for ds, k in CELLS:
        if not complete(ds, k):
            continue
        for cfg in CONFIGS:
            rr = [R[(cfg, ds, k, s)] for s in SEEDS]
            pl = "—" if LAM[cfg] == 0 else mean_sd([r["pres_last"] for r in rr], "{:.4f}")[0]
            L.append(f"| {ds} | {k} | {cfg} | {mean_sd([100 * r['bav'] for r in rr])[0]} | {mean_sd([100 * r['tab'] for r in rr])[0]} | "
                     f"{mean_sd([100 * r['teg'].mean() for r in rr])[0]} | {mean_sd([100 * r['F'].mean() for r in rr])[0]} | {pl} |")
        cands = [c for c in CONFIGS if LAM[c] > 0]
        score = {c: np.mean([R[(c, ds, k, s)]["bav"] for s in SEEDS]) for c in cands}
        sel[(ds, k)] = sorted(cands, key=lambda c: (-score[c], LAM[c], 0 if c.endswith("_nb") else 1))[0]
    L.append("")

    # ---- table 2
    L.append("## 표 2 — 선택 설정 비교 (sel = λ > 0 6개 중 원본 best_acc_valid seed 평균 최고; 동률이면 작은 λ, 그다음 nb)")
    L.append("| 데이터셋 | shot | sel | TEG+P(sel) | TEG+P(base) | D | F(base) | TEG+P(sel) − TEG+P(base) | TEG+P(sel) − D | TEG+P(sel) − F(base) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    K = {"K0": 0, "K1": 0, "K2": 0}
    p55 = {}
    for ds, k in CELLS:
        if (ds, k) not in sel:
            continue
        c = sel[(ds, k)]
        S = [R[(c, ds, k, s)] for s in SEEDS]
        B = [R[("base", ds, k, s)] for s in SEEDS]
        t0, m0, se0 = paired([100 * (a["teg"] - b["teg"]) for a, b in zip(S, B)])
        tD, mD, _ = paired([100 * (a["teg"] - b["D"]) for a, b in zip(S, B)])
        t1, m1, se1 = paired([100 * (a["teg"] - b["F"]) for a, b in zip(S, B)])
        K["K0"] += m0 > 0 and m0 > 2 * se0
        K["K1"] += m1 > 0 and m1 > 2 * se1
        p55[(ds, k)] = mD
        L.append(f"| {ds} | {k} | {c} | {mean_sd([100 * r['teg'].mean() for r in S])[0]} | {mean_sd([100 * r['teg'].mean() for r in B])[0]} | "
                 f"{mean_sd([100 * r['D'].mean() for r in B])[0]} | {mean_sd([100 * r['F'].mean() for r in B])[0]} | {t0} | {tD} | {t1} |")
    L.append("")

    # ---- table 3
    L.append("## 표 3 — 결합 위에서의 비교")
    L.append("| 데이터셋 | shot | sel | F(sel) | F(base) | F(sel) − F(base) | F(sel) − TEG+P(sel) |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        if (ds, k) not in sel:
            continue
        c = sel[(ds, k)]
        S = [R[(c, ds, k, s)] for s in SEEDS]
        B = [R[("base", ds, k, s)] for s in SEEDS]
        t2, m2, se2 = paired([100 * (a["F"] - b["F"]) for a, b in zip(S, B)])
        K["K2"] += m2 > 0 and m2 > 2 * se2
        L.append(f"| {ds} | {k} | {c} | {mean_sd([100 * r['F'].mean() for r in S])[0]} | {mean_sd([100 * r['F'].mean() for r in B])[0]} | {t2} | "
                 f"{paired([100 * (a['F'] - a['teg']) for a in S])[0]} |")
    L.append("")

    # ---- table 4
    L.append("## 표 4 — 표현 진단 (코사인 프로토타입 probe, 고정 test 에피소드, %)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"emb·{c}" for c in CONFIGS) + " | Â¹X (행 정규화) |")
    L.append("|---|---|" + "---|" * (len(CONFIGS) + 1))
    for ds, k in CELLS:
        if not complete(ds, k):
            continue
        cols = [mean_sd([100 * R[(c, ds, k, s)]["probe_emb"].mean() for s in SEEDS])[0] for c in CONFIGS]
        cols.append(mean_sd([100 * R[("base", ds, k, s)]["probe_h1"].mean() for s in SEEDS])[0])
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")

    # ---- judgements
    L.append("## 판정 (§3.3, 값만)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| K0 TEG 대비 | 표 2 TEG+P(sel) − TEG+P(base): 평균 > 0 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {K['K0']}/6 |")
    L.append(f"| K1 단독 가치 | 표 2 TEG+P(sel) − F(base): 평균 > 0 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {K['K1']}/6 |")
    L.append(f"| K2 보완 가치 | 표 3 F(sel) − F(base): 평균 > 0 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {K['K2']}/6 |")
    L.append("")
    edge = [key for key, r in R.items() if r["F_edge"]]
    L.append(f"- F(cfg) 온도 격자 끝 최적값: {len(edge)}개 run {edge[:10] if edge else ''}")
    L.append("")

    n55 = sum(1 for v in p55.values() if v > 0)
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P54 | K0 ≥ 4/6 | K0 = {K['K0']}/6 | {'일치' if K['K0'] >= 4 else '어긋남'} |")
    L.append(f"| P55 | 표 2 TEG+P(sel) − D 평균 > 0 인 (데이터셋, shot) ≥ 3/6 | {n55}/6 (" + ", ".join(f"{p55[c]:+.2f}" for c in CELLS if c in p55) + f") | {'일치' if n55 >= 3 else '어긋남'} |")
    L.append(f"| P56 | K1 ≤ 2/6 | K1 = {K['K1']}/6 | {'일치' if K['K1'] <= 2 else '어긋남'} |")
    L.append(f"| P57 | K2 ≥ 3/6 | K2 = {K['K2']}/6 | {'일치' if K['K2'] >= 3 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}; runs={n_runs}; K={K}; sel={sel}")


if __name__ == "__main__":
    main()
