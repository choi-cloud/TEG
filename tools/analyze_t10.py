"""T10 analysis (seed 0-9, original reporting + fixed episodes) -> reports/T10_summary.md, results/T10/gate_checks.txt.

Configs: base, kl_h2_l1, kl_h2_l10 (pool nb). Original datasets seed 0-9; extra datasets (corafull, coauthorCS, ogbn-arxiv) seed 0-4.
sel per (dataset, shot): the kl_h2 lambda with the highest seed-mean original best_acc_valid (ties -> smaller lambda).
M0/M1/M2 counted over the 6 original (dataset, shot) cells; extra datasets reported in separate rows. No RNG.
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
RES = os.path.join(ROOT, "results", "T10")
R07 = os.path.join(ROOT, "results", "T07")
OUT = os.path.join(ROOT, "reports", "T10_summary.md")
CONFIGS = ["base", "kl_h2_l1", "kl_h2_l10"]
LAM = {"base": 0.0, "kl_h2_l1": 1.0, "kl_h2_l10": 10.0}
ORIG = ["Amazon_clothing", "dblp", "Amazon_electronics"]
EXTRA = ["corafull", "coauthorCS", "ogbn-arxiv"]
SHOTS = [1, 5]
SEEDS = {**{ds: list(range(10)) for ds in ORIG}, **{ds: list(range(5)) for ds in EXTRA}}
EP_KEYS = ("epoch", "ep_idx", "mode", "support", "query", "classes")


def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m, sd


def paired(diff_by_seed):
    d = np.concatenate(diff_by_seed)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    sg = (m > 0) - (m < 0)
    agree = sum(1 for x in diff_by_seed if sg != 0 and ((float(x.mean()) > 0) - (float(x.mean()) < 0)) == sg)
    return f"{m:+.2f} ± {se:.2f} ({np.mean(d > 0):.2f}, {agree}/{len(diff_by_seed)})", m, se


def run_dir(cfg, ds, k, s):
    return os.path.join(RES, cfg, f"{ds}_5w{k}s", f"seed{s}")


def main():
    R, missing = {}, []
    gates = {"Ga": [], "Gb": [], "Gc": []}
    for ds in ORIG + EXTRA:
        H2 = None
        for k in SHOTS:
            for s in SEEDS[ds]:
                ref = None
                for cfg in CONFIGS:
                    d = run_dir(cfg, ds, k, s)
                    if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy")):
                        missing.append((cfg, ds, k, s))
                        continue
                    if H2 is None:
                        H2 = fb.diffusion_features(ds)
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
                    if LAM[cfg] > 0 and not all(e.get("pres_loss_mean") is not None and math.isfinite(e["pres_loss_mean"]) for e in run["epochs"]):
                        gates["Gc"].append((cfg, ds, k, s))
                    npz = os.path.join(d, "fixed_eval_logits.npz")
                    Zt = dict(np.load(npz))
                    m = np.where(Zt["mode"] == 1)[0]
                    m = m[np.argsort(Zt["ep_idx"][m], kind="stable")]
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    R[(cfg, ds, k, s)] = {"bav": run["final"]["best_acc_valid"], "tab": run["final"]["test_acc_at_best_valid"],
                                          "teg": (Zt["logits"][m].argmax(-1) == Zt["query_y"][m]).mean(-1), "F": ev["acc"], "D": ev["acc_D"]}
            print(ds, k, "done", flush=True)
        del H2

    def seeds_ok(ds, k):
        return [s for s in SEEDS[ds] if all((c, ds, k, s) in R for c in CONFIGS)]

    sel = {}
    for ds in ORIG + EXTRA:
        for k in SHOTS:
            ss = seeds_ok(ds, k)
            if ss:
                cands = ["kl_h2_l1", "kl_h2_l10"]
                score = {c: np.mean([R[(c, ds, k, s)]["bav"] for s in ss]) for c in cands}
                sel[(ds, k)] = sorted(cands, key=lambda c: (-score[c], LAM[c]))[0]

    L = ["# T10 요약 — seed 10개 확대 + 원 보고 방식 병기 (+ 추가 데이터셋)", ""]
    L.append(f"- run 수: **{len(R)} / 270** (기존 3 데이터셋 × 3 설정 × {{1,5}}-shot × seed 0–9 = 180, 추가 3 데이터셋 × 3 설정 × {{1,5}}-shot × seed 0–4 = 90). 누락: {missing or '없음'}.")
    L.append("- 추가 데이터셋: `dataset/`에 있고 `utils.load_data`가 코드 변경 없이 읽는 corafull, coauthorCS, ogbn-arxiv(오프라인 로딩 확인). `dataset/` 목록: Amazon_clothing, Amazon_electronics, CiteSeer, cora, CS, dblp, ogbn_arxiv(CiteSeer는 `load_data` 미지원).")
    L.append("- sel = kl_h2 λ ∈ {1, 10} 중 원본 best_acc_valid seed 평균 최고(동률이면 작은 λ), (데이터셋, shot)마다 해당 seed 전체 기준. M0–M2는 기존 3 데이터셋 6셀 기준.")
    L.append("")
    L.append("## 표 1 — 원 보고 방식 (`test_acc_at_best_valid`, %)")
    L.append("| 데이터셋 | shot | seed 수 | sel | TEG 평균 ± sd | TEG 95% CI | TEG+보존(sel) 평균 ± sd | TEG+보존 95% CI | seed 단위 짝지은 차 평균 ± SE (부호 일치) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    M = {"M0": 0, "M1": 0, "M2": 0}
    for ds in ORIG + EXTRA:
        for k in SHOTS:
            ss = seeds_ok(ds, k)
            if not ss:
                continue
            c = sel[(ds, k)]
            a = np.array([100 * R[("base", ds, k, s)]["tab"] for s in ss])
            b = np.array([100 * R[(c, ds, k, s)]["tab"] for s in ss])
            n = len(ss)
            tcrit = stats.t.ppf(0.975, n - 1)
            ci = lambda x: f"[{x.mean() - tcrit * x.std(ddof=1) / math.sqrt(n):.2f}, {x.mean() + tcrit * x.std(ddof=1) / math.sqrt(n):.2f}]"
            d = b - a
            md, sed = d.mean(), d.std(ddof=1) / math.sqrt(n)
            agree = int(np.sum(np.sign(d) == np.sign(md))) if md != 0 else 0
            if ds in ORIG:
                M["M0"] += md > 0 and md > 2 * sed
            L.append(f"| {ds} | {k} | {n} | {c} | {mean_sd(list(a))[0]} | {ci(a)} | {mean_sd(list(b))[0]} | {ci(b)} | {md:+.2f} ± {sed:.2f} ({agree}/{n}) |")
    L.append("")
    L.append("## 표 2 — 고정 에피소드 (seed당 test 200, 짝지은 차는 에피소드 단위 평균 ± SE)")
    L.append("| 데이터셋 | shot | seed 수 | TEG | D | F(base) | TEG+보존(sel) | F(sel) | TEG+P − TEG | TEG+P − D | TEG+P − F(base) | F(sel) − F(base) | F(sel) − TEG+P |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in ORIG + EXTRA:
        for k in SHOTS:
            ss = seeds_ok(ds, k)
            if not ss:
                continue
            c = sel[(ds, k)]
            B = [R[("base", ds, k, s)] for s in ss]
            S = [R[(c, ds, k, s)] for s in ss]
            acc = lambda rr, key: mean_sd([100 * r[key].mean() for r in rr])[0]
            d1 = paired([100 * (x["teg"] - y["teg"]) for x, y in zip(S, B)])
            dD = paired([100 * (x["teg"] - y["D"]) for x, y in zip(S, B)])
            dF = paired([100 * (x["teg"] - y["F"]) for x, y in zip(S, B)])
            dFF = paired([100 * (x["F"] - y["F"]) for x, y in zip(S, B)])
            dFT = paired([100 * (x["F"] - x["teg"]) for x in S])
            if ds in ORIG:
                M["M1"] += dF[1] > 0 and dF[1] > 2 * dF[2]
                M["M2"] += dFF[1] > 0 and dFF[1] > 2 * dFF[2]
            L.append(f"| {ds} | {k} | {len(ss)} | {acc(B, 'teg')} | {acc(B, 'D')} | {acc(B, 'F')} | {acc(S, 'teg')} | {acc(S, 'F')} | {d1[0]} | {dD[0]} | {dF[0]} | {dFF[0]} | {dFT[0]} |")
    L.append("")
    L.append("## 판정 (기존 3 데이터셋 6셀)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| M0 | 표 1 TEG+보존(sel) − TEG, seed 단위 짝지은 차 평균 > 0 이고 > 2·SE | {M['M0']}/6 |")
    L.append(f"| M1 | 표 2 TEG+P(sel) − F(base) 평균 > 0 이고 > 2·SE (T07 K1 정의) | {M['M1']}/6 |")
    L.append(f"| M2 | 표 2 F(sel) − F(base) 평균 > 0 이고 > 2·SE (T07 K2 정의) | {M['M2']}/6 |")
    L.append("")
    L.append("## 참고 — 기존 3 데이터셋 seed 0–2 `base` vs T07 `base` (`test_acc_at_best_valid`, 일치 요구 없음)")
    L.append("| 데이터셋 | shot | seed | T10 | T07 |")
    L.append("|---|---|---|---|---|")
    same = 0
    for ds in ORIG:
        for k in SHOTS:
            for s in range(3):
                if ("base", ds, k, s) not in R:
                    continue
                t07 = json.load(open(os.path.join(R07, "base", f"{ds}_5w{k}s", f"seed{s}", "run.json")))["final"]["test_acc_at_best_valid"]
                same += t07 == R[("base", ds, k, s)]["tab"]
                L.append(f"| {ds} | {k} | {s} | {R[('base', ds, k, s)]['tab']:.4f} | {t07:.4f} |")
    L.append(f"- 같은 값: {same}/18")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P62 | M0 = 6/6 (기존 3 데이터셋) | M0 = {M['M0']}/6 | {'일치' if M['M0'] == 6 else '어긋남'} |")
    L.append(f"| P63 | M1 ≤ 3/6 | M1 = {M['M1']}/6 | {'일치' if M['M1'] <= 3 else '어긋남'} |")
    L.append(f"| P64 | M2 ≥ 3/6 | M2 = {M['M2']}/6 | {'일치' if M['M2'] >= 3 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    G = [f"complete: {len(R)}/270; missing {missing or 'none'}",
         f"Ga episodes (eval_logits all epochs + fixed) identical across 3 configs: mismatches {gates['Ga'] or 'none'}",
         f"Gb checkpoint re-evaluation at ckpt_epoch 50/50: failures {gates['Gb'] or 'none'}",
         f"Gc pres_loss_mean finite (lambda > 0): failures {gates['Gc'] or 'none'}", f"sel: {sel}", f"M: {M}"]
    open(os.path.join(RES, "gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main()
