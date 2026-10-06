"""T08b analysis: T08 tables recomputed with view-L score = log(dumped probability) -> reports/T08b_summary.md.

All computation goes through tools/fusion_baseline.py. Inputs: results/T08/**/{eval_logits.npz, run.json}.
Analysis RNG: query half split random.Random(5000 + seed) (one instance per run, same split as T08).
Table formatting of A-D and J1/J2 follows tools/analyze_t08.py so that l_transform="prob" output can be compared
line by line with reports/T08_summary.md (gate G3, tools/check_t08b.py).
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_baseline as fb  # noqa: E402

ROOT = fb.ROOT
R08 = os.path.join(ROOT, "results", "T08")
OUT = os.path.join(ROOT, "reports", "T08b_summary.md")

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
N_WAY, N_QRY = 5, 5
A21 = fb.A21
A5 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
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


def run_dir(ds, k, s):
    return os.path.join(R08, f"{ds}_5w{k}s", f"seed{s}")


def compute_runs(l_transform, H_by_ds=None):
    """Per-run quantities for tables A-D (+ alpha = 0.5) under the given view-L transform."""
    runs = {}
    for ds in DATASETS:
        H = H_by_ds[ds] if H_by_ds is not None else fb.diffusion_features(ds)
        for k in SHOTS:
            for s in SEEDS:
                r = fb.prepare_run(run_dir(ds, k, s), H, l_transform)
                yv, yt = r["targets"]["valid"]["y"], r["targets"]["test"]["y"]
                a_val = fb.select_alpha(*r["lp"]["valid"], yv)
                lpL_t, lpD_t = r["lp"]["test"]
                curve = {float(a): fb.episode_acc(fb.combine(lpL_t, lpD_t, a), yt) for a in A21}
                # table C: 12/13 query split per test episode, alpha in A5, both directions
                rng = random.Random(5000 + s)
                adapt, fixed = [], []
                for i in range(len(yt)):
                    h1 = sorted(rng.sample(range(N_WAY * N_QRY), 12))
                    h2 = [q for q in range(N_WAY * N_QRY) if q not in h1]
                    accs = {}
                    for a in list(A5) + [a_val]:
                        pred = fb.combine(lpL_t[i], lpD_t[i], a).argmax(-1)
                        accs[float(a)] = ((pred[h1] == yt[i][h1]).mean(), (pred[h2] == yt[i][h2]).mean())

                    def pick(half):
                        top = max(accs[float(a)][half] for a in A5)
                        cands = [float(a) for a in A5 if accs[float(a)][half] == top]
                        return min(cands, key=lambda a: (abs(a - a_val), -a))

                    adapt.append((accs[pick(0)][1] + accs[pick(1)][0]) / 2)
                    fixed.append((accs[a_val][0] + accs[a_val][1]) / 2)
                cL = lpL_t.argmax(-1) == yt
                cD = lpD_t.argmax(-1) == yt
                runs[(ds, k, s)] = {
                    "e": r["targets"]["epoch"], "TL": r["T_L"], "s": r["s"], "iT": r["iT"], "iS": r["iS"], "a_val": a_val,
                    "curve": curve, "acc05": fb.episode_acc(fb.combine(lpL_t, lpD_t, 0.5), yt),
                    "adapt": np.array(adapt), "fixed": np.array(fixed), "n_replaced": r["n_replaced"],
                    "argmax_same": bool(np.array_equal(r["lL"]["test"].argmax(-1), r["targets"]["test"]["prob"].argmax(-1))
                                        and np.array_equal(r["lL"]["valid"].argmax(-1), r["targets"]["valid"]["prob"].argmax(-1))),
                    "D": {"L만": float((cL & ~cD).mean()), "D만": float((~cL & cD).mean()), "둘 다 정답": float((cL & cD).mean()),
                          "둘 다 오답": float((~cL & ~cD).mean()), "하나라도 정답": float((cL | cD).mean())},
                }
    return runs


def render(runs):
    """Tables A, B, C, D and J1/J2 in T08's format. Returns (sections dict of line lists, derived values)."""
    sec, val = {}, {"gain": {}, "better": {}, "diff_better": {}, "diff_teg": {}, "a_val_mean": {}, "TL_mean": {},
                    "acc_val": {}, "c_diff": {}, "curve_mean": {}}
    L = ["| 데이터셋 | shot | " + " | ".join(f"{a:.2f}" for a in A21) + " | acc(α=0) | acc(α=1) | max | argmax α | max − max(끝점) |",
         "|---|---|" + "---|" * (len(A21) + 5)]
    for ds, k in CELLS:
        per_a = {float(a): [100 * runs[(ds, k, s)]["curve"][float(a)].mean() for s in SEEDS] for a in A21}
        means = {a: sum(v) / len(v) for a, v in per_a.items()}
        val["curve_mean"][(ds, k)] = means
        mx = max(means.values())
        am = max(a for a, v in means.items() if v == mx)
        gain = mx - max(means[0.0], means[1.0])
        val["gain"][(ds, k)] = gain
        L.append(f"| {ds} | {k} | " + " | ".join(f"{means[float(a)]:.2f}" for a in A21) +
                 f" | {mean_sd(per_a[0.0])[0]} | {mean_sd(per_a[1.0])[0]} | {mx:.2f} | {am:.2f} | {gain:+.2f} |")
    sec["A"] = L

    L = ["| 데이터셋 | shot | α_val (seed순) | α_val 평균 | T_L (seed순) | s (seed순) | test acc(α_val) | 차 vs TEG(α=1) | 차 vs D(α=0) | 나은 끝점 | 차 vs 나은 끝점 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    j1 = 0
    for ds, k in CELLS:
        rr = [runs[(ds, k, s)] for s in SEEDS]
        acc_val = [r["curve"][r["a_val"]] for r in rr]
        d1 = [100 * (a - r["curve"][1.0]) for a, r in zip(acc_val, rr)]
        d0 = [100 * (a - r["curve"][0.0]) for a, r in zip(acc_val, rr)]
        better = 1.0 if val["curve_mean"][(ds, k)][1.0] >= val["curve_mean"][(ds, k)][0.0] else 0.0
        val["better"][(ds, k)] = better
        db = [100 * (a - r["curve"][better]) for a, r in zip(acc_val, rr)]
        tb, mb, seb = paired(db)
        if mb > 0 and mb > 2 * seb:
            j1 += 1
        t1, m1, se1 = paired(d1)
        val["diff_teg"][(ds, k)] = (m1, se1)
        val["diff_better"][(ds, k)] = (tb, mb, seb)
        am = sum(r["a_val"] for r in rr) / len(rr)
        val["a_val_mean"][(ds, k)] = am
        val["TL_mean"][(ds, k)] = sum(r["TL"] for r in rr) / len(rr)
        acc_txt = mean_sd([100 * a.mean() for a in acc_val])[0]
        val["acc_val"][(ds, k)] = acc_txt
        av_txt = ", ".join("{:.2f}".format(r["a_val"]) for r in rr)
        tl_txt = ", ".join("{:g}".format(r["TL"]) for r in rr)
        s_txt = ", ".join("{:g}".format(r["s"]) for r in rr)
        L.append(f"| {ds} | {k} | {av_txt} | {am:.3f} | {tl_txt} | "
                 f"{s_txt} | {acc_txt} | {t1} | {paired(d0)[0]} | "
                 f"{'TEG(α=1)' if better == 1.0 else 'D(α=0)'} | {tb} |")
    sec["B"] = L

    L = ["| 데이터셋 | shot | acc(적응) | acc(고정 α_val) | 적응 − 고정 | J2 조건 충족 |", "|---|---|---|---|---|---|"]
    j2 = 0
    for ds, k in CELLS:
        rr = [runs[(ds, k, s)] for s in SEEDS]
        dd = [100 * (r["adapt"] - r["fixed"]) for r in rr]
        txt, m, se = paired(dd)
        val["c_diff"][(ds, k)] = txt
        ok = m >= 1.0 and m > 2 * se
        j2 += ok
        L.append(f"| {ds} | {k} | {mean_sd([100 * r['adapt'].mean() for r in rr])[0]} | {mean_sd([100 * r['fixed'].mean() for r in rr])[0]} | {txt} | {'예' if ok else '아니오'} |")
    sec["C"] = L

    keys = ["L만", "D만", "둘 다 정답", "둘 다 오답", "하나라도 정답"]
    L = ["| 데이터셋 | shot | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for ds, k in CELLS:
        rr = [runs[(ds, k, s)] for s in SEEDS]
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd([100 * r["D"][key] for r in rr])[0] for key in keys) + " |")
    sec["D"] = L

    sec["J"] = ["| 판정 | 정의 | 값 |", "|---|---|---|",
                f"| J1 결합 가치 | 표 B 차 vs 나은 끝점: 평균 > 0 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {j1}/6 |",
                f"| J2 task별 판단 가치 | 표 C 적응 − 고정: 평균 ≥ 1.0%p 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {j2}/6 |"]
    val["J1"], val["J2"] = j1, j2
    return sec, val


def main():
    H_by_ds = {ds: fb.diffusion_features(ds) for ds in DATASETS}
    runs_log = compute_runs("log", H_by_ds)
    runs_prob = compute_runs("prob", H_by_ds)
    sec, val = render(runs_log)
    sec_p, val_p = render(runs_prob)

    Lo = ["# T08b 요약 — E0 재분석: 관점 L 점수 = log(덤프 확률)", ""]
    Lo.append("- run 수: 18 / 18 (T08 run 재사용, 학습 재실행 없음). 대상: run마다 `best_epoch_valid`의 valid 50·test 50 에피소드.")
    Lo.append("- 계산: `tools/fusion_baseline.py`, `l_transform=\"log\"`. 정확도 %, seed 3개 평균 ± sd. 짝지은 차: 에피소드 150개 평균 ± SE (양수 비율, seed 부호 일치 수/3), %p.")
    Lo.append(f"- log(0) 대체 원소 수(대상 valid·test 전체, 18 run 합): {sum(r['n_replaced'] for r in runs_log.values())}")
    Lo.append("")
    Lo += ["## 표 A — 결합 곡선 (test, α ∈ A21). max·argmax는 test로 고른 값(상한 참고용)"] + sec["A"] + [""]
    Lo += ["## 표 B — valid로 고른 α (α_val = argmin valid NLL, 동률이면 1에 가까운 값)"] + sec["B"] + [""]

    Lo.append("## 표 B2 — α = 0.5 고정 (T_L, s는 valid 보정)")
    Lo.append("| 데이터셋 | shot | test acc(α=0.5) | 차 vs TEG(α=1) | 차 vs D(α=0) | 차 vs 나은 끝점 | 차 vs α_val 결합 | \\|acc(0.5) − acc(α_val)\\| (seed 평균 기준) |")
    Lo.append("|---|---|---|---|---|---|---|---|")
    p53 = {}
    for ds, k in CELLS:
        rr = [runs_log[(ds, k, s)] for s in SEEDS]
        b = val["better"][(ds, k)]
        d = lambda ref: paired([100 * (r["acc05"] - ref(r)) for r in rr])[0]
        gap = abs(np.mean([100 * r["acc05"].mean() for r in rr]) - np.mean([100 * r["curve"][r["a_val"]].mean() for r in rr]))
        p53[(ds, k)] = gap
        Lo.append(f"| {ds} | {k} | {mean_sd([100 * r['acc05'].mean() for r in rr])[0]} | {d(lambda r: r['curve'][1.0])} | {d(lambda r: r['curve'][0.0])} | "
                  f"{d(lambda r: r['curve'][b])} | {d(lambda r: r['curve'][r['a_val']])} | {gap:.2f} |")
    Lo.append("")
    Lo += ["## 표 C — task별 선택의 여지 (test query 12/13 반분할 1회, α ∈ A5, 양방향 평균)"] + sec["C"] + [""]
    Lo += ["## 표 D — query 단위 상보성 (test, 결합 전 단독 예측, 비율 %)"] + sec["D"] + [""]

    Lo.append("## 표 E — T08(관점 L = 확률) 대비 T08b(관점 L = log 확률)")
    Lo.append("T08 열은 `fusion_baseline.py`의 `l_transform=\"prob\"` 결과다(T08 요약과 같은 값인지는 게이트 G3에서 확인).")
    Lo.append("")
    Lo.append("| 데이터셋 | shot | α_val 평균 (T08 / T08b) | T_L 평균 (T08 / T08b) | test acc(α_val) (T08 / T08b) | 차 vs 나은 끝점 (T08 / T08b) | 표 A max − max(끝점) (T08 / T08b) | 표 C 적응 − 고정 (T08 / T08b) |")
    Lo.append("|---|---|---|---|---|---|---|---|")
    for c in CELLS:
        Lo.append(f"| {c[0]} | {c[1]} | {val_p['a_val_mean'][c]:.3f} / {val['a_val_mean'][c]:.3f} | {val_p['TL_mean'][c]:.4g} / {val['TL_mean'][c]:.4g} | "
                  f"{val_p['acc_val'][c]} / {val['acc_val'][c]} | {val_p['diff_better'][c][0]} / {val['diff_better'][c][0]} | "
                  f"{val_p['gain'][c]:+.2f} / {val['gain'][c]:+.2f} | {val_p['c_diff'][c]} / {val['c_diff'][c]} |")
    Lo.append(f"| J1 | | | | | | | {val_p['J1']}/6 / {val['J1']}/6 |")
    Lo.append(f"| J2 | | | | | | | {val_p['J2']}/6 / {val['J2']}/6 |")
    Lo.append("")
    Lo += ["## 판정 (T08 §3.3 정의, log 기준)"] + sec["J"] + [""]

    edge = [((ds, k, s), "T_L", r["TL"]) for (ds, k, s), r in runs_log.items() if r["iT"] in (0, len(fb.TL_GRID) - 1)] + \
           [((ds, k, s), "s", r["s"]) for (ds, k, s), r in runs_log.items() if r["iS"] in (0, len(fb.S_GRID) - 1)]
    Lo.append(f"- 격자 끝 최적값(log 기준): {edge or '없음'}")
    Lo.append(f"- argmax(log p) = argmax(p): {sum(r['argmax_same'] for r in runs_log.values())}/18 run (대상 valid·test 모든 query)")
    Lo.append("")

    n51 = sum(1 for c in CELLS if val["diff_teg"][c][0] > 2 * val["diff_teg"][c][1])
    n53 = sum(1 for v in p53.values() if v <= 0.5)
    Lo.append("## 사전 등록 대비")
    Lo.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    Lo.append("|---|---|---|---|")
    Lo.append(f"| P50 | J2 ≤ 1/6 | J2 = {val['J2']}/6 | {'일치' if val['J2'] <= 1 else '어긋남'} |")
    Lo.append(f"| P51 | 표 B 결합(α_val) − TEG 평균 > 2·SE 인 (데이터셋, shot) = 6/6 | {n51}/6 (" +
              ", ".join(f"{val['diff_teg'][c][0]:+.2f}±{val['diff_teg'][c][1]:.2f}" for c in CELLS) + f") | {'일치' if n51 == 6 else '어긋남'} |")
    Lo.append(f"| P52 | J1 ≥ 3/6 | J1 = {val['J1']}/6 | {'일치' if val['J1'] >= 3 else '어긋남'} |")
    Lo.append(f"| P53 | 표 B2 \\|acc(α=0.5) − acc(α_val)\\| ≤ 0.5%p 인 (데이터셋, shot) ≥ 5/6 | {n53}/6 (" +
              ", ".join(f"{p53[c]:.2f}" for c in CELLS) + f") | {'일치' if n53 >= 5 else '어긋남'} |")
    open(OUT, "w").write("\n".join(Lo) + "\n")
    print(f"wrote {OUT}")
    return runs_log, runs_prob, sec, sec_p


if __name__ == "__main__":
    main()
