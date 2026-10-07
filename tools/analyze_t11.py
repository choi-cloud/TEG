"""T11 analysis (Rehearse 2x2) -> reports/T11_summary.md, results/T11/gate_checks.txt.

Configs: base, P (kl_h2 lambda 1, nb), R (reh_p 0.5, beta 1), P+R. Fixed episodes (test 200 per seed). No RNG.
Pseudo-class alignment (table 3, metric only): per pseudo class the fraction of members carrying its most frequent real label, averaged.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_baseline as fb  # noqa: E402

ROOT = fb.ROOT
RES = os.path.join(ROOT, "results", "T11")
OUT = os.path.join(ROOT, "reports", "T11_summary.md")
CONFIGS = ["base", "P", "R", "P+R"]
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
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


def run_dir(cfg, ds, k, s):
    return os.path.join(RES, cfg, f"{ds}_5w{k}s", f"seed{s}")


def main():
    R, missing = {}, []
    gates = {"Ga": [], "Gb": [], "Gc": [], "fake": []}
    for ds in DATASETS:
        H2 = fb.diffusion_features(ds)
        for k in SHOTS:
            for s in SEEDS:
                ref, fake_ref = None, None
                for cfg in CONFIGS:
                    d = run_dir(cfg, ds, k, s)
                    need = ["run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy"] + (["fake_classes.npz"] if "R" in cfg else [])
                    if not all(os.path.exists(os.path.join(d, f)) for f in need):
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
                    for key in (["pres_loss_mean"] if "P" in cfg else []) + (["reh_loss_mean"] if "R" in cfg else []):
                        if not all(e.get(key) is not None and math.isfinite(e[key]) for e in run["epochs"]):
                            gates["Gc"].append((cfg, ds, k, s, key))
                    rec = {}
                    if "R" in cfg:
                        fc = np.load(os.path.join(d, "fake_classes.npz"))
                        if fake_ref is None:
                            fake_ref = (fc["members"], fc["offsets"])
                        elif not (np.array_equal(fake_ref[0], fc["members"]) and np.array_equal(fake_ref[1], fc["offsets"])):
                            gates["fake"].append((cfg, ds, k, s))
                        labels = np.load(os.path.join(d, "labels.npy"))
                        test_cls = set(int(c) for c in json.load(open(os.path.join(d, "split.json")))["class_list_test"])
                        purity, test_frac = [], []
                        for a, b in zip(fc["offsets"][:-1], fc["offsets"][1:]):
                            lab = labels[fc["members"][a:b]].astype(int)
                            purity.append(np.bincount(lab).max() / len(lab))
                            test_frac.append(np.mean([l in test_cls for l in lab]))
                        st = run["rehearse"]
                        rec = {"n_fake": st["n_fake"], "n_clusters": st["n_clusters"], "size_med": float(np.median(st["fake_sizes"])),
                               "purity": float(np.mean(purity)), "test_frac": float(np.mean(test_frac)), "reh_steps": st["reh_steps_total"]}
                    npz = os.path.join(d, "fixed_eval_logits.npz")
                    Zt = dict(np.load(npz))
                    m = np.where(Zt["mode"] == 1)[0]
                    m = m[np.argsort(Zt["ep_idx"][m], kind="stable")]
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    R[(cfg, ds, k, s)] = {"tab": run["final"]["test_acc_at_best_valid"], "bav": run["final"]["best_acc_valid"],
                                          "teg": (Zt["logits"][m].argmax(-1) == Zt["query_y"][m]).mean(-1), "F": ev["acc"], **rec}
            print(ds, k, "done", flush=True)
        del H2

    complete = lambda ds, k: all((c, ds, k, s) in R for c in CONFIGS for s in SEEDS)
    L = ["# T11 요약 — Rehearse(가짜 클래스 에피소드) 2×2", ""]
    L.append(f"- run 수: **{len(R)} / 72** (4 설정 × 3 데이터셋 × 5-way {{1,5}}-shot × seed 0–2). 누락: {missing or '없음'}.")
    L.append("- 설정: base / P = kl_h2 λ = 1(nb) / R = reh_p 0.5, β 1 / P+R. 정확도 %, seed 평균 ± sd. 짝지은 차: 고정 test 에피소드 600개 평균 ± SE (양수 비율, seed 부호 일치 수/3), %p.")
    L.append("")
    L.append("## 표 1 — 고정 test 정확도와 F(cfg)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"{c} 고정 test" for c in CONFIGS) + " | " + " | ".join(f"F({c})" for c in CONFIGS) + " |")
    L.append("|---|---|" + "---|" * (2 * len(CONFIGS)))
    for ds, k in CELLS:
        if not complete(ds, k):
            continue
        cols = [mean_sd([100 * R[(c, ds, k, s)]["teg"].mean() for s in SEEDS])[0] for c in CONFIGS]
        cols += [mean_sd([100 * R[(c, ds, k, s)]["F"].mean() for s in SEEDS])[0] for c in CONFIGS]
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")
    L.append("## 표 2 — 짝지은 차")
    L.append("| 데이터셋 | shot | R − base | P − base | (P+R) − P | 상호작용 [(P+R) − P] − [R − base] |")
    L.append("|---|---|---|---|---|---|")
    J = {"R1": 0, "R2": 0}
    for ds, k in CELLS:
        if not complete(ds, k):
            continue
        g = lambda c, s: R[(c, ds, k, s)]["teg"]
        r_b = paired([100 * (g("R", s) - g("base", s)) for s in SEEDS])
        p_b = paired([100 * (g("P", s) - g("base", s)) for s in SEEDS])
        pr_p = paired([100 * (g("P+R", s) - g("P", s)) for s in SEEDS])
        inter = paired([100 * ((g("P+R", s) - g("P", s)) - (g("R", s) - g("base", s))) for s in SEEDS])
        J["R1"] += pr_p[1] > 0 and pr_p[1] > 2 * pr_p[2]
        J["R2"] += r_b[1] > 0 and r_b[1] > 2 * r_b[2]
        L.append(f"| {ds} | {k} | {r_b[0]} | {p_b[0]} | {pr_p[0]} | {inter[0]} |")
    L.append("")
    L.append("## 표 3 — 가짜 클래스 통계 (R 설정 run 기준, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 클러스터 수 | 가짜 클래스 수 | 크기 중앙값 | 다수 실제 클래스 비율 평균 | 구성원 중 test 클래스 노드 비율 | 가짜 에피소드 step 수(전 에폭) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        rr = [R[("R", ds, k, s)] for s in SEEDS if ("R", ds, k, s) in R]
        if not rr:
            continue
        L.append(f"| {ds} | {k} | {mean_sd([r['n_clusters'] for r in rr], '{:.1f}')[0]} | {mean_sd([r['n_fake'] for r in rr], '{:.1f}')[0]} | "
                 f"{mean_sd([r['size_med'] for r in rr], '{:.1f}')[0]} | {mean_sd([r['purity'] for r in rr], '{:.3f}')[0]} | "
                 f"{mean_sd([r['test_frac'] for r in rr], '{:.3f}')[0]} | {mean_sd([r['reh_steps'] for r in rr], '{:.1f}')[0]} |")
    L.append("")
    L.append("## 판정")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| R1 | (P+R) − P 평균 > 0 이고 > 2·SE 인 (데이터셋, shot) 수 | {J['R1']}/6 |")
    L.append(f"| R2 | R − base 같은 기준 | {J['R2']}/6 |")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P65 | R1 ≤ 2/6 | R1 = {J['R1']}/6 | {'일치' if J['R1'] <= 2 else '어긋남'} |")
    L.append(f"| P66 | R2 ≥ 3/6 | R2 = {J['R2']}/6 | {'일치' if J['R2'] >= 3 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    G = [f"complete: {len(R)}/72; missing {missing or 'none'}",
         f"Ga episodes (eval_logits all epochs + fixed) identical across 4 configs: mismatches {gates['Ga'] or 'none'}",
         f"Gb checkpoint re-evaluation at ckpt_epoch 50/50: failures {gates['Gb'] or 'none'}",
         f"Gc pres_loss_mean / reh_loss_mean finite at every epoch: failures {gates['Gc'] or 'none'}",
         f"pseudo classes identical between R and P+R: mismatches {gates['fake'] or 'none'}", f"J: {J}"]
    open(os.path.join(RES, "gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main()
