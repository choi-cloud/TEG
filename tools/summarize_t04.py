"""Read results/T04/**/run.json and write reports/T04_summary.md (tables 1-4, appendix, P34-P36)."""
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "T04")
OUT = os.path.join(ROOT, "reports", "T04_summary.md")

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ["off", "A_0.1", "A_0.3", "A_0.5", "B_0.1", "B_0.3", "B_0.5"]
SEL = {"A_sel": ["A_0.1", "A_0.3", "A_0.5"], "B_sel": ["B_0.1", "B_0.3", "B_0.5"]}


def load_runs():
    runs = {}
    for path in glob.glob(os.path.join(RESULTS, "*_5w*s", "seed*", "run.json")):
        run = json.load(open(path))
        cfg = run["config"]
        eps = [json.loads(l) for l in open(os.path.join(os.path.dirname(path), "episodes.jsonl"))]
        runs[(cfg["dataset"], cfg["shot"], cfg["seed"])] = (run, eps)
    return runs


def select(run, names):
    # valid-selected beta per seed: highest best_valid_acc; ties -> smaller beta (list order)
    mc = run["mem_configs"]
    best = max(names, key=lambda n: (mc[n]["best_valid_acc"], -names.index(n)))
    return best, mc[best]["test_acc_at_best_valid"]


def paired(arm, off):
    d = [100 * (a - o) for a, o in zip(arm, off)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    sign = (mean > 0) - (mean < 0)
    agree = sum(1 for x in d if sign != 0 and ((x > 0) - (x < 0)) == sign)
    detect = n > 1 and abs(mean) > 2 * se and agree >= math.ceil(0.8 * n)
    meaningful = abs(mean) >= 1.0
    return mean, se, agree, n, detect, meaningful


def fmt_cell(arm, off):
    mean_acc = 100 * sum(arm) / len(arm)
    m, se, agree, n, detect, meaningful = paired(arm, off)
    marks = ("†" if detect else "") + ("*" if meaningful else "")
    return f"{mean_acc:.2f} ({m:+.2f} ± {se:.2f}, {agree}/{n}){marks}", m


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "null"
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{m:.4f} ± {sd:.4f}", m


def main():
    runs = load_runs()
    status = [json.loads(l) for l in open(os.path.join(RESULTS, "_runs.jsonl"))] if os.path.exists(os.path.join(RESULTS, "_runs.jsonl")) else []
    complete_seeds = [s for s in SEEDS if all((ds, k, s) in runs for ds in DATASETS for k in SHOTS)]
    seeds = complete_seeds
    L = []
    L.append("# T04 요약 — 3 데이터셋 × 5-way {1,5}-shot × seed, `--mem`")
    L.append("")
    L.append(f"- 완료 seed 수: **{len(seeds)}** ({', '.join(map(str, seeds))}). run.json 수: {len(runs)} / 30.")
    L.append("- 지표: 설정별 `test_acc_at_best_valid` (TEG 원본 동점 규칙). 단위 %.")
    L.append("- 셀: \"seed 평균 정확도 (off 대비 짝지은 차 mean ± SE, 부호 일치 수/seed 수)\". 차 단위 %p. SE = sd(d)/√n.")
    L.append("- 표시: † 검출(|mean d| > 2·SE 이고 부호 일치 ≥ 4/5), * 의미(|mean d| ≥ 1.0 %p).")
    L.append("")

    # Table 1
    L.append("## 표 1 — 정확도")
    L.append("| 데이터셋 | shot | " + " | ".join(CONFIGS) + " |")
    L.append("|---|---|" + "---|" * len(CONFIGS))
    for ds in DATASETS:
        for k in SHOTS:
            off = [runs[(ds, k, s)][0]["mem_configs"]["off"]["test_acc_at_best_valid"] for s in seeds]
            cells = [f"{100 * sum(off) / len(off):.2f}"]
            for c in CONFIGS[1:]:
                arm = [runs[(ds, k, s)][0]["mem_configs"][c]["test_acc_at_best_valid"] for s in seeds]
                cells.append(fmt_cell(arm, off)[0])
            L.append(f"| {ds} | {k} | " + " | ".join(cells) + " |")
    L.append("")

    # Table 1b: one beta per arm, chosen by the mean over the 6 (dataset, shot) rows of mean d
    def test_acc(c, ds, k):
        return [runs[(ds, k, s)][0]["mem_configs"][c]["test_acc_at_best_valid"] for s in seeds]

    rows = [(ds, k) for ds in DATASETS for k in SHOTS]
    avg_d = {c: sum(paired(test_acc(c, ds, k), test_acc("off", ds, k))[0] for ds, k in rows) / len(rows) for c in CONFIGS[1:]}
    best = {arm: max(names, key=lambda n: (avg_d[n], -names.index(n))) for arm, names in SEL.items()}
    L.append("## 표 1b — 설정별 대표 β (off / A-* / B-*)")
    L.append("arm마다 β 3개 중 \"6개 (데이터셋, shot)의 mean d 평균\"이 가장 큰 β를 하나 고른다(동점이면 작은 β). "
             "**test 결과로 고른 사후 선택**이다(seed별 valid 선택은 표 2).")
    L.append("")
    L.append("| β | " + " | ".join(CONFIGS[1:]) + " |")
    L.append("|---|" + "---|" * (len(CONFIGS) - 1))
    L.append("| 6행 평균 mean d (%p) | " + " | ".join(f"{avg_d[c]:+.3f}" for c in CONFIGS[1:]) + " |")
    L.append("")
    L.append(f"| 데이터셋 | shot | off | A-* ({best['A_sel']}) | B-* ({best['B_sel']}) |")
    L.append("|---|---|---|---|---|")
    for ds, k in rows:
        off = test_acc("off", ds, k)
        cells = [f"{100 * sum(off) / len(off):.2f}"] + [fmt_cell(test_acc(best[arm], ds, k), off)[0] for arm in SEL]
        L.append(f"| {ds} | {k} | " + " | ".join(cells) + " |")
    L.append("")

    # Table 2
    L.append("## 표 2 — valid로 고른 β (A_sel, B_sel)")
    L.append("seed마다 해당 arm의 β 3개 중 `best_valid_acc`가 가장 높은 β를 고르고(동점이면 작은 β), 그 β의 `test_acc_at_best_valid`를 쓴다.")
    L.append("")
    L.append("| 데이터셋 | shot | off | A_sel | 고른 β (seed순) | B_sel | 고른 β (seed순) |")
    L.append("|---|---|---|---|---|---|---|")
    sel_means = defaultdict(dict)
    for ds in DATASETS:
        for k in SHOTS:
            off = [runs[(ds, k, s)][0]["mem_configs"]["off"]["test_acc_at_best_valid"] for s in seeds]
            row = [f"{100 * sum(off) / len(off):.2f}"]
            for sel, names in SEL.items():
                picks = [select(runs[(ds, k, s)][0], names) for s in seeds]
                cell, m = fmt_cell([p[1] for p in picks], off)
                sel_means[sel][(ds, k)] = m
                row += [cell, ", ".join(p[0].split("_")[1] for p in picks)]
            L.append(f"| {ds} | {k} | " + " | ".join(row) + " |")
    L.append("")

    # Table 3
    L.append("## 표 3 — 전제 진단 (off의 `best_valid_epoch`에서, seed 평균 ± sd)")
    L.append("`mean_phat_same`·`mean_phat_diff`는 그 에폭 test 에피소드 50개 값의 평균(에피소드마다 same/diff 간선 수가 같아 전체 간선 평균과 같음).")
    L.append("")
    L.append("| 데이터셋 | shot | off best_valid_epoch (seed순) | auc_phat_pooled | auc_dist_pooled | phat − dist | mean_phat_same | mean_phat_diff |")
    L.append("|---|---|---|---|---|---|---|---|")
    p35 = {}
    for ds in DATASETS:
        for k in SHOTS:
            cols = defaultdict(list)
            epochs = []
            for s in seeds:
                run, eps = runs[(ds, k, s)]
                e = run["mem_configs"]["off"]["best_valid_epoch"]
                epochs.append(e)
                rec = run["epochs"][e]
                cols["phat"].append(rec["auc_phat_pooled"])
                cols["dist"].append(rec["auc_dist_pooled"])
                cols["diff"].append(None if rec["auc_phat_pooled"] is None else rec["auc_phat_pooled"] - rec["auc_dist_pooled"])
                te = [r for r in eps if r["epoch"] == e and r["mode"] == "test"]
                for key in ("mean_phat_same", "mean_phat_diff"):
                    vals = [r[key] for r in te]
                    cols[key].append(None if None in vals else sum(vals) / len(vals))
            out = {c: mean_sd(cols[c]) for c in ("phat", "dist", "diff", "mean_phat_same", "mean_phat_diff")}
            p35[(ds, k)] = (out["phat"], out["dist"])
            L.append(f"| {ds} | {k} | {', '.join(map(str, epochs))} | " + " | ".join(
                (v if isinstance(v, str) else v[0]) for v in out.values()) + " |")
    L.append("")

    # Table 4
    L.append("## 표 4 — seed별 원값 (`test_acc_at_best_valid`, %)")
    L.append("| 데이터셋 | shot | seed | " + " | ".join(CONFIGS) + " | A_sel β | B_sel β |")
    L.append("|---|---|---|" + "---|" * (len(CONFIGS) + 2))
    for ds in DATASETS:
        for k in SHOTS:
            for s in SEEDS:
                if (ds, k, s) not in runs:
                    L.append(f"| {ds} | {k} | {s} | " + " | ".join(["—"] * (len(CONFIGS) + 2)) + " |")
                    continue
                run = runs[(ds, k, s)][0]
                vals = [f"{100 * run['mem_configs'][c]['test_acc_at_best_valid']:.2f}" for c in CONFIGS]
                picks = [select(run, names)[0].split("_")[1] for names in SEL.values()]
                L.append(f"| {ds} | {k} | {s} | " + " | ".join(vals + picks) + " |")
    L.append("")

    # P34-P36
    def obs_count(sel):
        return sum(1 for v in sel_means[sel].values() if abs(v) < 1.0)

    n_cells = len(sel_means["A_sel"])
    p35_obs = sum(1 for v in p35.values() if not isinstance(v[0], str) and not isinstance(v[1], str) and v[0][1] < v[1][1])
    pa, pb = obs_count("A_sel"), obs_count("B_sel")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P34 | A_sel − off: 6개 모두 \\|mean d\\| < 1.0 %p | {pa}/{n_cells} | {'일치' if pa == 6 else '어긋남'} |")
    L.append(f"| P35 | 6개 모두 `auc_phat_pooled` < `auc_dist_pooled` (seed 평균) | {p35_obs}/{len(p35)} | {'일치' if p35_obs == 6 else '어긋남'} |")
    L.append(f"| P36 | B_sel − off: 6개 중 4개 이상 \\|mean d\\| < 1.0 %p | {pb}/{n_cells} | {'일치' if pb >= 4 else '어긋남'} |")
    L.append("")

    # Appendix
    L.append("## 부록 — 실패·재시도, run별 소요 시간")
    failed = [r for r in status if not r["ok"]]
    retried = [r for r in status if len(r["attempts"]) > 1]
    L.append(f"- 실패(재시도 후): {len(failed)} — " + (", ".join(f"{r['dataset']} 5w{r['shot']}s seed{r['seed']}" for r in failed) or "없음"))
    L.append(f"- 재시도 발생: {len(retried)} — " + (", ".join(f"{r['dataset']} 5w{r['shot']}s seed{r['seed']}" for r in retried) or "없음"))
    L.append("")
    L.append("| 데이터셋 | shot | seed | 물리 GPU | 시도 | 프로세스 (s) | `wall_time_sec` (s) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sorted(status, key=lambda r: (r["seed"], DATASETS.index(r["dataset"]), r["shot"])):
        key = (r["dataset"], r["shot"], r["seed"])
        wt = f"{runs[key][0]['wall_time_sec']:.1f}" if key in runs else "—"
        L.append(f"| {r['dataset']} | {r['shot']} | {r['seed']} | {r['gpu']} | {len(r['attempts'])} | {r['attempts'][-1]['sec']:.1f} | {wt} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
