"""T22 analysis (seed 0-9: TEG vs InfoNCE with pool nb vs pool all) -> reports/T22_summary.md, results/T22/gate_checks.txt.

Copied from tools/analyze_t10r.py; configs replaced (base, infonce_nb lambda {10, 30, 100}, infonce_all lambda {10, 30, 100}),
tables / judgements replaced by T22 §4-§5. Only seeds with all 7 configs complete are used for a (dataset, shot). lambda per
(pool, dataset, shot) by the seed-mean original best_acc_valid (ties -> smaller lambda). Judgements K1-K6 on the fixed
episodes (test 200 per seed). Gf: |T22 infonce_nb seed mean - T10r infonce seed mean| <= 2.0 %p per cell (fixed test,
T10r selected lambda from reports/T10r_summary.md 표 1). Judgements and the preregistration block are written only when Gf
passes and --no_verdict is not given. No RNG.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_baseline as fb  # noqa: E402

ROOT = fb.ROOT
RES = os.path.join(ROOT, "results", "T22")
R10 = os.path.join(ROOT, "results", "T10r")
OUT = os.path.join(ROOT, "reports", "T22_summary.md")
GRID = [10.0, 30.0, 100.0]
POOLS = ["nb", "all"]
NAME = "infonce_{}_l{:g}"
CONFIGS = ["base"] + [NAME.format(p, l) for p in POOLS for l in GRID]
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(10))
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
EP_KEYS = ("epoch", "ep_idx", "mode", "support", "query", "classes")


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


def seed_paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
    agree = int(np.sum(np.sign(d) == np.sign(m))) if m != 0 else 0
    return f"{m:+.2f} ± {se:.2f} ({agree}/{n})", m, se


def ci95(x):
    x = np.asarray(x)
    n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return f"[{x.mean() - h:.2f}, {x.mean() + h:.2f}]"


def run_dir(cfg, ds, k, s, root=RES):
    return os.path.join(root, cfg, f"{ds}_5w{k}s", f"seed{s}")


def lam_of(cfg):
    return 0.0 if cfg == "base" else float(cfg.rsplit("_l", 1)[1])


def pool_of(cfg):
    return "nb" if cfg == "base" else cfg.split("_")[1]


def fixed_teg(npz):
    Zt = dict(np.load(npz))
    m = np.where(Zt["mode"] == 1)[0]
    m = m[np.argsort(Zt["ep_idx"][m], kind="stable")]
    return (Zt["logits"][m].argmax(-1) == Zt["query_y"][m]).mean(-1)


def t10r_selected():
    """reports/T10r_summary.md 표 1: (ds, k) -> InfoNCE selected lambda."""
    lines = open(os.path.join(ROOT, "reports", "T10r_summary.md")).read().split("\n")
    i = lines.index("## 표 1 — 원 보고 방식 (`test_acc_at_best_valid`)")
    out = {}
    for ln in lines[i + 3 : i + 9]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        out[(c[0], int(c[1]))] = float(c[3].split("/")[1].replace("‡", "").strip())
    return out


def main(no_verdict=False):
    R, missing = {}, []
    gates = {"Gb": [], "Gc": [], "Gd": []}
    for ds in DATASETS:
        H2 = fb.diffusion_features(ds)
        for k in SHOTS:
            for s in SEEDS:
                ref = None
                for cfg in CONFIGS:
                    d = run_dir(cfg, ds, k, s)
                    if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "eval_logits.npz", "fixed_eval_logits.npz")):
                        missing.append((cfg, ds, k, s))
                        continue
                    run = json.load(open(os.path.join(d, "run.json")))
                    Z, Fz = np.load(os.path.join(d, "eval_logits.npz")), np.load(os.path.join(d, "fixed_eval_logits.npz"))
                    cur = {("e", x): Z[x] for x in EP_KEYS} | {("f", x): Fz[x] for x in EP_KEYS}
                    if ref is None:
                        ref = cur
                    elif not all(np.array_equal(ref[x], cur[x]) for x in ref):
                        gates["Gb"].append((cfg, ds, k, s))
                    ck = run["ckpt_recheck"]
                    if not (ck and ck["epoch"] == run["ckpt_epoch"] and ck["n"] == 50 and ck["n_equal"] == 50):
                        gates["Gc"].append(((cfg, ds, k, s), ck))
                    c = run["config"]
                    ok = c["pres_pool"] == pool_of(cfg) and float(c["pres_lambda"]) == lam_of(cfg)
                    if lam_of(cfg) > 0:
                        ok = ok and c["pres_loss"] == "infonce" and all(
                            e.get("pres_loss_mean") is not None and math.isfinite(e["pres_loss_mean"]) for e in run["epochs"])
                    if not ok:
                        gates["Gd"].append((cfg, ds, k, s))
                    npz = os.path.join(d, "fixed_eval_logits.npz")
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    R[(cfg, ds, k, s)] = {"bav": run["final"]["best_acc_valid"], "tab": run["final"]["test_acc_at_best_valid"],
                                          "teg": fixed_teg(npz), "F": ev["acc"], "D": ev["acc_D"],
                                          "F_edge": ev["iT"] in (0, len(fb.TL_GRID) - 1) or ev["iS"] in (0, len(fb.S_GRID) - 1)}
            print(ds, k, "done", flush=True)
        del H2

    seeds_ok = {c: [s for s in SEEDS if all((cfg, *c, s) in R for cfg in CONFIGS)] for c in CELLS}
    sel = {}
    for (ds, k), ss in seeds_ok.items():
        if not ss:
            continue
        for p in POOLS:
            cands = [NAME.format(p, l) for l in GRID]
            score = {c: np.mean([R[(c, ds, k, s)]["bav"] for s in ss]) for c in cands}
            sel[(p, ds, k)] = sorted(cands, key=lambda c: (-score[c], lam_of(c)))[0]
    edge = lambda p, ds, k: lam_of(sel[(p, ds, k)]) in (GRID[0], GRID[-1])

    # ---- Gf / T5: reproduction against T10r (fixed test, T10r selected lambda)
    sel10 = t10r_selected()
    T5, gf = {}, {}
    for ds, k in CELLS:
        lam = sel10[(ds, k)]
        a = [100 * R[(NAME.format("nb", lam), ds, k, s)]["teg"].mean() for s in seeds_ok[(ds, k)]]
        b = [100 * fixed_teg(os.path.join(run_dir(f"infonce_l{lam:g}", ds, k, s, R10), "fixed_eval_logits.npz")).mean() for s in seeds_ok[(ds, k)]]
        T5[(ds, k)] = (lam, a, b)
        gf[(ds, k)] = abs(np.mean(a) - np.mean(b))
    gf_pass = all(v <= 2.0 for v in gf.values())
    verdict = gf_pass and not no_verdict

    L = ["# T22 요약 — 대조(InfoNCE) 풀 nb vs all, seed 0–9", ""]
    L.append(f"- run 수: **{len(R)} / 420** (7 설정 × 3 데이터셋 × 5-way {{1,5}}-shot × seed 0–9). 누락: {missing or '없음'}.")
    L.append("- (데이터셋, shot)별 사용 seed: " + ", ".join(f"{ds} {k}: {len(seeds_ok[(ds, k)])}" for ds, k in CELLS) + " (7개 설정이 모두 완료된 seed만).")
    L.append("- TEG = `base`, C_nb = 선택 λ의 `infonce_nb`, C_all = 선택 λ의 `infonce_all`. 선택 λ = 풀별 원본 best_acc_valid seed 평균 최고(동률이면 작은 λ). "
             "격자 {10, 30, 100}. ‡ = 선택 λ가 격자 끝.")
    L.append("- 정확도 %. D·F(cfg)는 `fusion_baseline.evaluate_npz`(α = 0.5, log). 고정 에피소드 짝지은 차: 평균 ± SE (양수 비율, seed 부호 일치/seed 수).")
    L.append("")
    L.append("## T1 — 원 보고 방식 (`test_acc_at_best_valid`, seed 평균 ± sd, 95% CI)")
    L.append("| 데이터셋 | shot | n | 선택 λ (nb / all) | TEG | TEG 95% CI | C_nb | C_nb 95% CI | C_all | C_all 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        ss = seeds_ok[(ds, k)]
        t = [100 * R[("base", ds, k, s)]["tab"] for s in ss]
        a = [100 * R[(sel[("nb", ds, k)], ds, k, s)]["tab"] for s in ss]
        b = [100 * R[(sel[("all", ds, k)], ds, k, s)]["tab"] for s in ss]
        lam = " / ".join(f"{lam_of(sel[(p, ds, k)]):g}{'‡' if edge(p, ds, k) else ''}" for p in POOLS)
        L.append(f"| {ds} | {k} | {len(ss)} | {lam} | {mean_sd(t)[0]} | {ci95(t)} | {mean_sd(a)[0]} | {ci95(a)} | {mean_sd(b)[0]} | {ci95(b)} |")
    L.append("")

    L.append("## T2 — 고정 에피소드 (seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | TEG | D | F(base) | C_nb | C_all | F(C_nb) | F(C_all) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    keys = ["C_all − TEG", "C_all − C_nb", "C_all − D", "C_all − F(base)", "F(C_all) − F(base)", "F(C_all) − F(C_nb)"]
    diffs = {}
    for ds, k in CELLS:
        ss = seeds_ok[(ds, k)]
        B = [R[("base", ds, k, s)] for s in ss]
        N = [R[(sel[("nb", ds, k)], ds, k, s)] for s in ss]
        A = [R[(sel[("all", ds, k)], ds, k, s)] for s in ss]
        acc = lambda rr, key: mean_sd([100 * r[key].mean() for r in rr])[0]
        L.append(f"| {ds} | {k} | {acc(B, 'teg')} | {acc(B, 'D')} | {acc(B, 'F')} | {acc(N, 'teg')} | {acc(A, 'teg')} | {acc(N, 'F')} | {acc(A, 'F')} |")
        pd = lambda X, xk, Y, yk: paired([100 * (x[xk] - y[yk]) for x, y in zip(X, Y)])
        diffs[(ds, k)] = {"C_all − TEG": pd(A, "teg", B, "teg"), "C_all − C_nb": pd(A, "teg", N, "teg"), "C_all − D": pd(A, "teg", B, "D"),
                          "C_all − F(base)": pd(A, "teg", B, "F"), "F(C_all) − F(base)": pd(A, "F", B, "F"),
                          "F(C_all) − F(C_nb)": pd(A, "F", N, "F")}
    L.append("")
    L.append("## T3 — 고정 에피소드 짝지은 차 (%p)")
    L.append("| 데이터셋 | shot | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for c, dd in diffs.items():
        L.append(f"| {c[0]} | {c[1]} | " + " | ".join(dd[key][0] for key in keys) + " |")
    L.append("")

    L.append("## T4 — λ 곡선 (seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 풀 | λ | n | 원본 best_acc_valid | 고정 test |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for p in POOLS:
            for l in GRID:
                c = NAME.format(p, l)
                ss = [s for s in SEEDS if (c, ds, k, s) in R]
                rr = [R[(c, ds, k, s)] for s in ss]
                L.append(f"| {ds} | {k} | {p} | {l:g}{' (선택)' if sel[(p, ds, k)] == c else ''} | {len(ss)} | "
                         f"{mean_sd([100 * r['bav'] for r in rr])[0]} | {mean_sd([100 * r['teg'].mean() for r in rr])[0]} |")
    L.append("")

    L.append("## T5 — 재현 대조: T22 `infonce_nb` vs T10r `infonce` (고정 test, T10r 선택 λ)")
    L.append("| 데이터셋 | shot | λ | n | T22 nb seed 평균 | T10r seed 평균 | 차 (Gf) | seed 단위 차 평균 ± SE (부호 일치/n) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (ds, k), (lam, a, b) in T5.items():
        L.append(f"| {ds} | {k} | {lam:g} | {len(a)} | {np.mean(a):.2f} | {np.mean(b):.2f} | {np.mean(a) - np.mean(b):+.2f} | {seed_paired(a, b)[0]} |")
    L.append(f"- Gf: 6셀 모두 \\|차\\| ≤ 2.0 %p → {'충족' if gf_pass else '미충족'} (최대 {max(gf.values()):.2f}).")
    L.append("")

    K = {}
    clear = lambda key: sum(1 for c in CELLS if diffs[c][key][1] > 0 and diffs[c][key][1] > 2 * diffs[c][key][2])
    K["K1"] = clear("C_all − TEG")
    K["K2"] = clear("C_all − D")
    K["K2_pos"] = sum(1 for c in CELLS if diffs[c]["C_all − D"][1] > 0)
    K["K3"] = sum(1 for c in CELLS if abs(diffs[c]["C_all − C_nb"][1]) <= 1.0)
    K["K3_neg"] = sum(1 for c in CELLS if diffs[c]["C_all − C_nb"][1] < 0 and -diffs[c]["C_all − C_nb"][1] > 2 * diffs[c]["C_all − C_nb"][2])
    K["K4"] = clear("F(C_all) − F(base)")
    K["K5"] = clear("C_all − F(base)")
    K["K6"] = sum(1 for ds, k in CELLS if lam_of(sel[("nb", ds, k)]) == lam_of(sel[("all", ds, k)]))
    if verdict:
        L.append("## 판정 (고정 에피소드, 값만; 분명 = 평균 > 0이고 > 2·SE)")
        L.append("| 판정 | 정의 | 값 |")
        L.append("|---|---|---|")
        L.append(f"| K1 | C_all − TEG 분명히 양수 | {K['K1']}/6 |")
        L.append(f"| K2 | C_all − D 분명히 양수; 평균 > 0 (기록) | {K['K2']}/6; {K['K2_pos']}/6 |")
        L.append(f"| K3 | \\|C_all − C_nb\\| 평균 ≤ 1.0 %p; 분명히 음수 (기록) | {K['K3']}/6; {K['K3_neg']}/6 |")
        L.append(f"| K4 | F(C_all) − F(base) 분명히 양수 | {K['K4']}/6 |")
        L.append(f"| K5 (기록) | C_all − F(base) 분명히 양수 | {K['K5']}/6 |")
        L.append(f"| K6 (기록) | 선택 λ가 nb와 all에서 같은 셀 | {K['K6']}/6 |")
        L.append("")
        L.append("## 사전 등록 대조")
        L.append("| 예측 | 조건 | 관측 | 일치 |")
        L.append("|---|---|---|---|")
        for p, cond, obs, ok in (("P112", "K1 = 6/6", f"K1 = {K['K1']}/6", K["K1"] == 6),
                                 ("P113", "K2(분명) ≥ 4/6", f"K2 = {K['K2']}/6", K["K2"] >= 4),
                                 ("P114", "K3 ≥ 5/6", f"K3 = {K['K3']}/6", K["K3"] >= 5),
                                 ("P115", "K4 = 6/6", f"K4 = {K['K4']}/6", K["K4"] == 6)):
            L.append(f"| {p} | {cond} | {obs} | {'일치' if ok else '어긋남'} |")
        L.append("")
    else:
        L.append("## 판정")
        L.append(f"- 쓰지 않음 ({'Gf 미충족' if not gf_pass else '--no_verdict'}).")
        L.append("")
    fe = [key for key, r in R.items() if r["F_edge"]]
    L.append(f"- F(cfg) 온도 격자 끝 최적값: {len(fe)}개 run {fe[:6] if fe else ''}")
    open(OUT, "w").write("\n".join(L) + "\n")
    G = [f"complete: {len(R)}/420; missing {missing or 'none'}",
         f"Gb episodes (eval_logits all epochs + fixed) identical across 7 configs: mismatches {gates['Gb'] or 'none'}",
         f"Gc checkpoint re-evaluation at ckpt_epoch 50/50: failures {gates['Gc'] or 'none'}",
         f"Gd config.pres_pool / pres_lambda match the config name, pres_loss infonce and pres_loss_mean finite (lambda > 0): failures {gates['Gd'] or 'none'}",
         f"Gf |T22 nb mean - T10r mean| <= 2.0 per cell: {'pass' if gf_pass else 'FAIL'} " + json.dumps({f'{ds}|{k}': round(v, 3) for (ds, k), v in gf.items()}),
         f"sel: { {f'{p}|{ds}|{k}': lam_of(c) for (p, ds, k), c in sel.items()} }",
         f"verdict written: {verdict}; K: {K if verdict else 'withheld'}"]
    open(os.path.join(RES, "gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main(no_verdict="--no_verdict" in sys.argv[1:])
