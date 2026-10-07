"""T14 analysis (BSC E1: negative-batch sampler only) -> reports/T14_summary.md, results/T14/gate_checks.txt.

Configs: base (lambda 0), uni, emb_a{0.05,0.2}, bsc_a{0.05,0.2}, bsc_a0.2_ref (infonce, T10r-selected lambda), 3 datasets x
5-way {1,5}-shot x seed 0-4. alpha per (emb | bsc, dataset, shot) by the seed-mean original best_acc_valid (ties -> smaller alpha).
Fixed episodes: test 200 per seed (paired over 1,000). +D = tools/fusion_baseline.evaluate_npz (alpha = 0.5).
Representation diagnostics on the test group of emb_best.npy (row L2-normalized): kNN purity (top-10 cosine inside the test
group), erank_node and BW from tools/analyze_t13.group_metrics; group construction tools/analyze_t13.groups_of. No RNG.
"""
import json
import math
import os
import sys

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import fusion_baseline as fb  # noqa: E402
from analyze_t13 import group_metrics, groups_of  # noqa: E402

ROOT = fb.ROOT
RES = os.path.join(ROOT, "results", "T14")
OUT = os.path.join(ROOT, "reports", "T14_summary.md")
CONFIGS = ["base", "uni", "emb_a0.05", "emb_a0.2", "bsc_a0.05", "bsc_a0.2", "bsc_a0.2_ref"]
ALPHAS = {"emb": ["emb_a0.05", "emb_a0.2"], "bsc": ["bsc_a0.05", "bsc_a0.2"]}
DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = list(range(5))
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
EP_KEYS = ("epoch", "ep_idx", "mode", "support", "query", "classes")
DIAG = ["fn_rate", "same_proto_rate", "masked_frac", "jumps"]


def mean_sd(xs, fmt="{:.2f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—", None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def paired(diff_by_seed):
    d = np.concatenate(diff_by_seed)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    sg = (m > 0) - (m < 0)
    agree = sum(1 for x in diff_by_seed if sg != 0 and ((float(x.mean()) > 0) - (float(x.mean()) < 0)) == sg)
    return f"{m:+.2f} ± {se:.2f} ({agree}/{len(diff_by_seed)})", m, se


def run_dir(cfg, ds, k, s):
    return os.path.join(RES, cfg, f"{ds}_5w{k}s", f"seed{s}")


def knn_purity(Z, nodes, y, k=10, block=4096):
    V = torch.nn.functional.normalize(torch.from_numpy(Z[nodes]).double(), dim=1)
    yy = torch.from_numpy(y)
    out = []
    for s in range(0, len(nodes), block):
        sim = V[s : s + block] @ V.T
        r = torch.arange(sim.shape[0])
        sim[r, r + s] = float("-inf")
        nb = sim.topk(k, dim=1).indices
        out.append((yy[nb] == yy[s : s + block, None]).double().mean(1))
    return float(torch.cat(out).mean())


def main():
    R, missing = {}, []
    gates = {"Ga": [], "Gb": [], "Gc": []}
    rep = {}
    for ds in DATASETS:
        H2 = fb.diffusion_features(ds)
        for k in SHOTS:
            for s in SEEDS:
                ref = None
                for cfg in CONFIGS:
                    d = run_dir(cfg, ds, k, s)
                    if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy")):
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
                    if cfg != "base":
                        for e in run["epochs"]:
                            vals = [e.get("pres_loss_mean")]
                            if e["epoch"] >= 1:
                                vals += [e["bsc"][x] for x in ("fn_rate", "same_proto_rate", "masked_frac")]
                                if cfg != "uni":
                                    vals.append(e["bsc"]["jumps"])
                            if not all(v is not None and math.isfinite(v) for v in vals):
                                gates["Gc"].append((cfg, ds, k, s, e["epoch"]))
                    npz = os.path.join(d, "fixed_eval_logits.npz")
                    Zt = dict(np.load(npz))
                    m = np.where(Zt["mode"] == 1)[0]
                    m = m[np.argsort(Zt["ep_idx"][m], kind="stable")]
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    G, _, _ = groups_of(d)
                    emb = np.load(os.path.join(d, "emb_best.npy"))
                    gm = group_metrics(emb, *G["test"])
                    ep = run["epochs"]
                    diag = {}
                    if cfg != "base":
                        for x in DIAG:
                            v = [e["bsc"][x] for e in ep if e["epoch"] >= 1]
                            diag[x] = (v[0], v[-1], float(np.mean(v)) if all(t is not None for t in v) else None)
                    R[(cfg, ds, k, s)] = {"bav": run["final"]["best_acc_valid"], "tab": run["final"]["test_acc_at_best_valid"],
                                          "train": ep[run["ckpt_epoch"]]["train_acc"],
                                          "teg": (Zt["logits"][m].argmax(-1) == Zt["query_y"][m]).mean(-1), "F": ev["acc"], "D": ev["acc_D"],
                                          "purity": knn_purity(emb, *G["test"]), "erank": gm["erank_node"], "BW": gm["BW"], "diag": diag,
                                          "graph": [e["bsc"]["graph"] for e in ep if e["epoch"] >= 1] if cfg != "base" else None,
                                          "wall": run["wall_time_sec"]}
                    if cfg == "base":
                        rep.setdefault((ds, k), []).append(knn_purity(H2, *G["test"]))
            print(ds, k, "done", flush=True)
        del H2

    sel = {}
    for ds, k in CELLS:
        for v, cands in ALPHAS.items():
            score = {c: np.mean([R[(c, ds, k, s)]["bav"] for s in SEEDS]) for c in cands}
            sel[(v, ds, k)] = sorted(cands, key=lambda c: (-score[c], float(c.split("_a")[1])))[0]
    name = {"원본": lambda ds, k: "base", "대조": lambda ds, k: "uni", "근접(sel)": lambda ds, k: sel[("emb", ds, k)],
            "BSC(sel)": lambda ds, k: sel[("bsc", ds, k)], "BSC+심판": lambda ds, k: "bsc_a0.2_ref"}
    rr = lambda cfg, ds, k: [R[(cfg, ds, k, s)] for s in SEEDS]

    L = ["# T14 요약 — BSC E1: 음성 배치 선택만 바꾼 대조 학습", ""]
    L.append(f"- run 수: **{len(R)} / 210** (7 설정 × 3 데이터셋 × 5-way {{1,5}}-shot × seed 0–4). 누락: {missing or '없음'}.")
    L.append("- λ: (데이터셋, shot)마다 `reports/T10r_summary.md` 표 1의 `infonce` 선택 λ 고정. 원본 = base, 대조 = uni, 근접(sel) = emb α 선택, BSC(sel) = bsc α 선택, BSC+심판 = bsc_a0.2_ref.")
    L.append("- α 선택: 원본 `best_acc_valid` seed 평균 최고(동률이면 작은 α). 정확도 %. seed 평균 ± sd. 짝지은 차: 고정 test 에피소드(seed당 200, 총 1,000) 평균 ± SE (seed 부호 일치 수/5).")
    L.append("")
    L.append("## 선택 α")
    L.append("| 데이터셋 | shot | 근접(sel) | BSC(sel) |")
    L.append("|---|---|---|---|")
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | {sel[('emb', ds, k)]} | {sel[('bsc', ds, k)]} |")
    L.append("")

    L.append("## 표 1 — 정확도")
    L.append("### 고정 test 정확도 (TEG 출력)")
    L.append("| 데이터셋 | shot | " + " | ".join(CONFIGS) + " |")
    L.append("|---|---|" + "---|" * len(CONFIGS))
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd([100 * r["teg"].mean() for r in rr(c, ds, k)])[0] for c in CONFIGS) + " |")
    L.append("")
    L.append("### 원 보고 방식 (`test_acc_at_best_valid`)")
    L.append("| 데이터셋 | shot | " + " | ".join(CONFIGS) + " |")
    L.append("|---|---|" + "---|" * len(CONFIGS))
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd([100 * r["tab"] for r in rr(c, ds, k)])[0] for c in CONFIGS) + " |")
    L.append("")
    L.append("### +D 결합 F(cfg) (고정 test)")
    FN = ["대조", "근접(sel)", "BSC(sel)", "BSC+심판"]
    L.append("| 데이터셋 | shot | " + " | ".join(f"F({n})" for n in FN) + " |")
    L.append("|---|---|" + "---|" * len(FN))
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd([100 * r["F"].mean() for r in rr(name[n](ds, k), ds, k)])[0] for n in FN) + " |")
    L.append("")

    L.append("## 표 2 — 짝지은 차 (고정 test 에피소드, %p)")
    cmp_ = [("대조 − 원본", "대조", "teg", "원본", "teg"), ("BSC(sel) − 대조", "BSC(sel)", "teg", "대조", "teg"),
            ("근접(sel) − 대조", "근접(sel)", "teg", "대조", "teg"), ("BSC(sel) − 근접(sel)", "BSC(sel)", "teg", "근접(sel)", "teg"),
            ("bsc_a0.2_ref − bsc_a0.2", "BSC+심판", "teg", None, "teg"), ("F(BSC sel) − F(대조)", "BSC(sel)", "F", "대조", "F")]
    L.append("| 데이터셋 | shot | " + " | ".join(c[0] for c in cmp_) + " | bsc_a0.05 − emb_a0.05 | bsc_a0.2 − emb_a0.2 |")
    L.append("|---|---|" + "---|" * (len(cmp_) + 2))
    J = {"P78": 0, "B1": 0, "B2": 0, "B3": 0, "B4a": 0, "B4b": 0, "B5": 0, "B6": 0}
    sgn = {"B1": {}, "B6": {}}
    for ds, k in CELLS:
        cols = []
        for lab, a, ak, b, bk in cmp_:
            A = rr(name[a](ds, k), ds, k)
            B = rr("bsc_a0.2", ds, k) if b is None else rr(name[b](ds, k), ds, k)
            txt, m, se = paired([100 * (x[ak] - y[bk]) for x, y in zip(A, B)])
            cols.append(txt)
            dist = m > 0 and m > 2 * se
            if lab == "대조 − 원본":
                J["P78"] += dist
            if lab == "BSC(sel) − 대조":
                J["B1"] += dist
                sgn["B1"][(ds, k)] = np.sign(m)
            if lab == "BSC(sel) − 근접(sel)":
                J["B2"] += dist
            if lab == "bsc_a0.2_ref − bsc_a0.2":
                J["B3"] += dist
        for a in ("0.05", "0.2"):
            cols.append(paired([100 * (x["teg"] - y["teg"]) for x, y in zip(rr(f"bsc_a{a}", ds, k), rr(f"emb_a{a}", ds, k))])[0])
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")

    L.append("## 표 3 — 배치 진단 (seed 평균; e1 = 에폭 1, e10 = 에폭 10, 평균 = 에폭 1–10 평균)")
    L.append("| 데이터셋 | shot | 설정 | " + " | ".join(f"{x} {t}" for x in DIAG for t in ("e1", "e10", "평균")) + " |")
    L.append("|---|---|---|" + "---|" * 12)
    for ds, k in CELLS:
        for c in CONFIGS[1:]:
            cols = []
            for x in DIAG:
                for t in range(3):
                    vals = [r["diag"][x][t] for r in rr(c, ds, k)]
                    _, mm = mean_sd(vals, "{:.4f}")
                    cols.append("—" if mm is None else (f"{mm:.2f}" if x == "jumps" else f"{mm:.4f}"))
            L.append(f"| {ds} | {k} | {c} | " + " | ".join(cols) + " |")
        fa = {c: np.mean([r["diag"]["fn_rate"][2] for r in rr(c, ds, k)]) for c in ("bsc_a0.05", "bsc_a0.2", "bsc_a0.2_ref")}
        J["B4a"] += fa["bsc_a0.2"] > fa["bsc_a0.05"]
        J["B4b"] += fa["bsc_a0.2_ref"] < fa["bsc_a0.2"]
    L.append("")
    L.append("### 표 3 참고 — 근접 그래프 통계 (에폭 1–10, seed 평균)")
    L.append("| 데이터셋 | shot | 설정 | 그룹 수 | 그룹 크기 중앙값 | 이웃 없는 노드 수 | 무향 간선 수 |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for c in CONFIGS[1:]:
            g = [x for r in rr(c, ds, k) for x in r["graph"]]
            f = lambda key: f"{np.mean([x[key] for x in g]):.1f}" if key in g[0] else "—"
            L.append(f"| {ds} | {k} | {c} | {f('n_groups')} | {f('group_size_median')} | {f('n_isolated')} | {f('n_edges_undirected')} |")
    L.append("")

    L.append("## 표 4 — 표현 진단 (test 그룹, `emb_best.npy`, 행 L2 정규화, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | 설정 | kNN 순도 (k = 10) | erank_node | BW |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for c in CONFIGS:
            X = rr(c, ds, k)
            L.append(f"| {ds} | {k} | {c} | {mean_sd([r['purity'] for r in X], '{:.4f}')[0]} | {mean_sd([r['erank'] for r in X])[0]} | "
                     f"{mean_sd([r['BW'] for r in X], '{:.3f}')[0]} |")
        L.append(f"| {ds} | {k} | Â²X (학습 없음, 참고) | {mean_sd(rep[(ds, k)], '{:.4f}')[0]} | — | — |")
        d6 = np.mean([x["purity"] - y["purity"] for x, y in zip(rr(name["BSC(sel)"](ds, k), ds, k), rr("uni", ds, k))])
        J["B6"] += d6 > 0
        sgn["B6"][(ds, k)] = np.sign(d6)
    L.append("")

    L.append("## 표 5 — base 학습 정확도 (`epochs[ckpt_epoch].train_acc`, T12 표 1과 같은 정의, %)")
    L.append("| 데이터셋 | shot | 대조 train | BSC(sel) train | BSC(sel) − 대조 (seed 평균 ± sd) |")
    L.append("|---|---|---|---|---|")
    for ds, k in CELLS:
        a, b = rr("uni", ds, k), rr(name["BSC(sel)"](ds, k), ds, k)
        txt, mm = mean_sd([100 * (y["train"] - x["train"]) for x, y in zip(a, b)], "{:+.2f}")
        J["B5"] += abs(mm) < 1.0
        L.append(f"| {ds} | {k} | {mean_sd([100 * r['train'] for r in a])[0]} | {mean_sd([100 * r['train'] for r in b])[0]} | {txt} |")
    L.append("")
    agree84 = sum(sgn["B1"][c] == sgn["B6"][c] for c in CELLS)

    L.append("## 판정 (단위 (데이터셋, shot) 6개, 분명 = 평균 > 0 이고 > 2·SE)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| (P78) | 대조 − 원본 분명 | {J['P78']}/6 |")
    L.append(f"| B1 | BSC(sel) − 대조 분명 | {J['B1']}/6 |")
    L.append(f"| B2 | BSC(sel) − 근접(sel) 분명 | {J['B2']}/6 |")
    L.append(f"| B3 | bsc_a0.2_ref − bsc_a0.2 분명 | {J['B3']}/6 |")
    L.append(f"| B4a | 전 에폭 평균 fn_rate bsc_a0.2 > bsc_a0.05 | {J['B4a']}/6 |")
    L.append(f"| B4b | 전 에폭 평균 fn_rate bsc_a0.2_ref < bsc_a0.2 | {J['B4b']}/6 |")
    L.append(f"| B5 | \\|BSC(sel) − 대조\\| train 차 < 1.0%p | {J['B5']}/6 |")
    L.append(f"| B6 | kNN 순도 BSC(sel) > 대조 (seed 평균) | {J['B6']}/6 |")
    L.append(f"| (P84) | B6·B1 부호 일치 (seed 평균 차 부호) | {agree84}/6 |")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    yn = lambda b: "일치" if b else "어긋남"
    L.append(f"| P78 | 대조 − 원본 6/6 분명 | {J['P78']}/6 | {yn(J['P78'] == 6)} |")
    L.append(f"| P79 | B1 ≥ 3/6 | {J['B1']}/6 | {yn(J['B1'] >= 3)} |")
    L.append(f"| P80 | B2 ≥ 3/6 | {J['B2']}/6 | {yn(J['B2'] >= 3)} |")
    L.append(f"| P81 | B4a = 6/6, B4b = 6/6 | B4a {J['B4a']}/6, B4b {J['B4b']}/6 | {yn(J['B4a'] == 6 and J['B4b'] == 6)} |")
    L.append(f"| P82 | B3 ≥ 2/6 | {J['B3']}/6 | {yn(J['B3'] >= 2)} |")
    L.append(f"| P83 | B5 ≥ 5/6 | {J['B5']}/6 | {yn(J['B5'] >= 5)} |")
    L.append(f"| P84 | B6·B1 부호 일치 ≥ 5/6 | {agree84}/6 | {yn(agree84 >= 5)} |")
    open(OUT, "w").write("\n".join(L) + "\n")

    wall = {c: float(np.mean([R[(c, ds, k, s)]["wall"] for ds, k in CELLS for s in SEEDS])) for c in CONFIGS}
    G = [f"complete: {len(R)}/210; missing {missing or 'none'}",
         f"Ga episodes (eval_logits all epochs + fixed) identical across 7 configs: mismatches {gates['Ga'] or 'none'}",
         f"Gb checkpoint re-evaluation at ckpt_epoch 50/50: failures {gates['Gb'] or 'none'}",
         f"Gc pres_loss_mean (all epochs) and diagnostics (epochs 1-10) finite: failures {gates['Gc'] or 'none'}",
         f"sel {sel}", f"J {J} agree84 {agree84}", f"wall_time_sec mean by config {wall}"]
    open(os.path.join(RES, "gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main()
