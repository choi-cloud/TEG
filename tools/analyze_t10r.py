"""T10r analysis (seed 0-9: TEG vs KL preservation vs InfoNCE, extended lambda grid) -> reports/T10r_summary.md,
results/T10r/gate_checks.txt.

Configs: base, kl_h2 lambda {1, 10, 30}, infonce lambda {10, 30, 100} (pool nb), 3 datasets x 5-way {1,5}-shot x seed 0-9.
Only seeds with all 7 configs complete are used for a (dataset, shot). lambda per (variant, dataset, shot) by the seed-mean
original best_acc_valid (ties -> smaller lambda). Judgements N0-N3 on the fixed episodes (test 200 per seed). No RNG.
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
RES = os.path.join(ROOT, "results", "T10r")
R09 = os.path.join(ROOT, "results", "T09")
OUT = os.path.join(ROOT, "reports", "T10r_summary.md")
GRID = {"kl": [1.0, 10.0, 30.0], "infonce": [10.0, 30.0, 100.0]}
NAME = {"kl": "kl_h2_l{:g}", "infonce": "infonce_l{:g}"}
CONFIGS = ["base"] + [NAME[v].format(l) for v in GRID for l in GRID[v]]
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(10))
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
EP_KEYS = ("epoch", "ep_idx", "mode", "support", "query", "classes")
LABEL = {"kl": "KL", "infonce": "InfoNCE"}


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
    return f"{m:+.2f} ± {se:.2f} ({agree}/{n})"


def ci95(x):
    x = np.asarray(x)
    n = len(x)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return f"[{x.mean() - h:.2f}, {x.mean() + h:.2f}]"


def run_dir(cfg, ds, k, s):
    return os.path.join(RES, cfg, f"{ds}_5w{k}s", f"seed{s}")


def lam_of(cfg):
    return 0.0 if cfg == "base" else float(cfg.rsplit("_l", 1)[1])


def main():
    R, missing = {}, []
    gates = {"Ga": [], "Gb": [], "Gc": []}
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
                        gates["Ga"].append((cfg, ds, k, s))
                    ck = run["ckpt_recheck"]
                    if not (ck and ck["epoch"] == run["ckpt_epoch"] and ck["n"] == 50 and ck["n_equal"] == 50):
                        gates["Gb"].append(((cfg, ds, k, s), ck))
                    if lam_of(cfg) > 0 and not all(e.get("pres_loss_mean") is not None and math.isfinite(e["pres_loss_mean"]) for e in run["epochs"]):
                        gates["Gc"].append((cfg, ds, k, s))
                    npz = os.path.join(d, "fixed_eval_logits.npz")
                    Zt = dict(np.load(npz))
                    m = np.where(Zt["mode"] == 1)[0]
                    m = m[np.argsort(Zt["ep_idx"][m], kind="stable")]
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    R[(cfg, ds, k, s)] = {"bav": run["final"]["best_acc_valid"], "tab": run["final"]["test_acc_at_best_valid"],
                                          "teg": (Zt["logits"][m].argmax(-1) == Zt["query_y"][m]).mean(-1), "F": ev["acc"], "D": ev["acc_D"],
                                          "F_edge": ev["iT"] in (0, len(fb.TL_GRID) - 1) or ev["iS"] in (0, len(fb.S_GRID) - 1)}
            print(ds, k, "done", flush=True)
        del H2

    seeds_ok = {c: [s for s in SEEDS if all((cfg, *c, s) in R for cfg in CONFIGS)] for c in CELLS}
    sel = {}
    for (ds, k), ss in seeds_ok.items():
        if not ss:
            continue
        for v in GRID:
            cands = [NAME[v].format(l) for l in GRID[v]]
            score = {c: np.mean([R[(c, ds, k, s)]["bav"] for s in ss]) for c in cands}
            sel[(v, ds, k)] = sorted(cands, key=lambda c: (-score[c], lam_of(c)))[0]
    edge = lambda v, ds, k: lam_of(sel[(v, ds, k)]) in (GRID[v][0], GRID[v][-1])

    L = ["# T10r 요약 — seed 10개: TEG vs 보존(KL, Â²X) vs 대조 학습(InfoNCE), λ 격자 끝 확장", ""]
    L.append(f"- run 수: **{len(R)} / 420** (7 설정 × 3 데이터셋 × 5-way {{1,5}}-shot × seed 0–9). 누락: {missing or '없음'}.")
    L.append("- (데이터셋, shot)별 사용 seed: " + ", ".join(f"{ds} {k}: {len(seeds_ok[(ds, k)])}" for ds, k in CELLS) + " (7개 설정이 모두 완료된 seed만).")
    L.append("- 선택 λ = 원본 best_acc_valid seed 평균 최고(동률이면 작은 λ). 격자: KL {1, 10, 30}, InfoNCE {10, 30, 100}. ‡ = 선택 λ가 격자 끝(최소 또는 최대).")
    L.append("- 정확도 %. 표 1: seed 평균 ± sd, 95% CI(t), seed 단위 짝지은 차 평균 ± SE (부호 일치 수/seed 수). 표 2: 고정 test 에피소드(seed당 200) 짝지은 차 평균 ± SE (양수 비율, seed 부호 일치 수/seed 수).")
    L.append("")

    L.append("## 표 1 — 원 보고 방식 (`test_acc_at_best_valid`)")
    L.append("| 데이터셋 | shot | n | 선택 λ (KL / InfoNCE) | TEG | TEG 95% CI | KL(sel) | KL 95% CI | InfoNCE(sel) | InfoNCE 95% CI | KL − TEG | InfoNCE − TEG | KL − InfoNCE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        ss = seeds_ok[(ds, k)]
        if len(ss) < 2:
            continue
        t = [100 * R[("base", ds, k, s)]["tab"] for s in ss]
        a = [100 * R[(sel[("kl", ds, k)], ds, k, s)]["tab"] for s in ss]
        b = [100 * R[(sel[("infonce", ds, k)], ds, k, s)]["tab"] for s in ss]
        lam = f"{lam_of(sel[('kl', ds, k)]):g}{'‡' if edge('kl', ds, k) else ''} / {lam_of(sel[('infonce', ds, k)]):g}{'‡' if edge('infonce', ds, k) else ''}"
        L.append(f"| {ds} | {k} | {len(ss)} | {lam} | {mean_sd(t)[0]} | {ci95(t)} | {mean_sd(a)[0]} | {ci95(a)} | {mean_sd(b)[0]} | {ci95(b)} | "
                 f"{seed_paired(a, t)} | {seed_paired(b, t)} | {seed_paired(a, b)} |")
    L.append("")

    L.append("## 표 2 — 고정 에피소드")
    L.append("| 데이터셋 | shot | TEG | D | F(base) | KL(sel) | InfoNCE(sel) | F(KL) | F(InfoNCE) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    N = {"N0": 0, "N1": 0, "N2": 0, "N3": 0}
    diffs = {}
    for ds, k in CELLS:
        ss = seeds_ok[(ds, k)]
        if len(ss) < 2:
            continue
        B = [R[("base", ds, k, s)] for s in ss]
        K = [R[(sel[("kl", ds, k)], ds, k, s)] for s in ss]
        I = [R[(sel[("infonce", ds, k)], ds, k, s)] for s in ss]
        acc = lambda rr, key: mean_sd([100 * r[key].mean() for r in rr])[0]
        L.append(f"| {ds} | {k} | {acc(B, 'teg')} | {acc(B, 'D')} | {acc(B, 'F')} | {acc(K, 'teg')} | {acc(I, 'teg')} | {acc(K, 'F')} | {acc(I, 'F')} |")
        pd = lambda X, xk, Y, yk: paired([100 * (x[xk] - y[yk]) for x, y in zip(X, Y)])
        diffs[(ds, k)] = {"KL − TEG": pd(K, "teg", B, "teg"), "InfoNCE − TEG": pd(I, "teg", B, "teg"), "KL − InfoNCE": pd(K, "teg", I, "teg"),
                          "KL − F(base)": pd(K, "teg", B, "F"), "InfoNCE − F(base)": pd(I, "teg", B, "F"),
                          "F(KL) − F(base)": pd(K, "F", B, "F"), "F(InfoNCE) − F(base)": pd(I, "F", B, "F")}
        for key, nm in (("KL − TEG", "N0"), ("InfoNCE − TEG", "N1"), ("KL − InfoNCE", "N2")):
            _, mm, se = diffs[(ds, k)][key]
            N[nm] += mm > 0 and mm > 2 * se
        _, mm, se = pd(I, "teg", K, "teg")
        N["N3"] += mm > 0 and mm > 2 * se
    L.append("")
    keys = ["KL − TEG", "InfoNCE − TEG", "KL − InfoNCE", "KL − F(base)", "InfoNCE − F(base)", "F(KL) − F(base)", "F(InfoNCE) − F(base)"]
    L.append("### 표 2 — 짝지은 차 (%p)")
    L.append("| 데이터셋 | shot | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for c, dd in diffs.items():
        L.append(f"| {c[0]} | {c[1]} | " + " | ".join(dd[key][0] for key in keys) + " |")
    L.append("")

    L.append("## 표 3 — λ 곡선 (7개 설정, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 설정 | n | 원본 best_acc_valid | 원본 test_acc_at_best_valid | 고정 test |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for c in CONFIGS:
            ss = [s for s in SEEDS if (c, ds, k, s) in R]
            if len(ss) < 2:
                continue
            rr = [R[(c, ds, k, s)] for s in ss]
            L.append(f"| {ds} | {k} | {c} | {len(ss)} | {mean_sd([100 * r['bav'] for r in rr])[0]} | {mean_sd([100 * r['tab'] for r in rr])[0]} | "
                     f"{mean_sd([100 * r['teg'].mean() for r in rr])[0]} |")
    L.append("")

    L.append("## 판정 (고정 에피소드, 평균 > 0 이고 평균 > 2·SE인 (데이터셋, shot) 수)")
    L.append("| 판정 | 비교 | 값 |")
    L.append("|---|---|---|")
    for nm, txt in (("N0", "KL − TEG"), ("N1", "InfoNCE − TEG"), ("N2", "KL − InfoNCE"), ("N3", "InfoNCE − KL")):
        L.append(f"| {nm} | {txt} | {N[nm]}/6 |")
    L.append("")
    fe = [key for key, r in R.items() if r["F_edge"]]
    L.append(f"- F(cfg) 온도 격자 끝 최적값: {len(fe)}개 run {fe[:6] if fe else ''}")
    L.append("")

    # Gd: seeds 0-2 vs T09
    L.append("## 참고 — seed 0–2, T09 같은 설정과 병기 (`test_acc_at_best_valid`, 일치 요구 없음)")
    L.append("| 데이터셋 | shot | seed | 설정 | T10r | T09 |")
    L.append("|---|---|---|---|---|---|")
    same, tot = 0, 0
    for ds, k in CELLS:
        for s in range(3):
            for c in ("base", "kl_h2_l1", "kl_h2_l10", "infonce_l10"):
                p = os.path.join(R09, c, f"{ds}_5w{k}s", f"seed{s}", "run.json")
                if (c, ds, k, s) not in R or not os.path.exists(p):
                    continue
                t9 = json.load(open(p))["final"]["test_acc_at_best_valid"]
                tot += 1
                same += t9 == R[(c, ds, k, s)]["tab"]
                L.append(f"| {ds} | {k} | {s} | {c} | {R[(c, ds, k, s)]['tab']:.4f} | {t9:.4f} |")
    L.append(f"- 같은 값: {same}/{tot}")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P67 | N0 = 6/6 | N0 = {N['N0']}/6 | {'일치' if N['N0'] == 6 else '어긋남'} |")
    L.append(f"| P68 | N1 = 6/6 | N1 = {N['N1']}/6 | {'일치' if N['N1'] == 6 else '어긋남'} |")
    L.append(f"| P69 | N2 ≤ 1/6 | N2 = {N['N2']}/6 | {'일치' if N['N2'] <= 1 else '어긋남'} |")
    L.append(f"| P70 | N3 ≥ 2/6 | N3 = {N['N3']}/6 | {'일치' if N['N3'] >= 2 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    G = [f"complete: {len(R)}/420; missing {missing or 'none'}",
         f"Ga episodes (eval_logits all epochs + fixed) identical across 7 configs: mismatches {gates['Ga'] or 'none'}",
         f"Gb checkpoint re-evaluation at ckpt_epoch 50/50: failures {gates['Gb'] or 'none'}",
         f"Gc pres_loss_mean finite (lambda > 0): failures {gates['Gc'] or 'none'}",
         f"Gd seeds 0-2 vs T09: same {same}/{tot}", f"sel: {sel}", f"N: {N}"]
    open(os.path.join(RES, "gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main()
