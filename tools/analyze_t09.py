"""T09 analysis (preserve-loss ablation + self-supervised baseline) -> reports/T09_summary.md, results/T09/gate_checks.txt.

Inputs: results/T09/<cfg>/<ds>_5w<k>s/seed<s>/{run.json, eval_logits.npz, fixed_eval_logits.npz, emb_best.npy}.
Comparisons on the fixed episodes (test 200 per seed), shared by all configs of a (dataset, shot, seed).
lambda selected per (variant, dataset, shot) by seed-mean original best_acc_valid (ties -> smaller lambda). No RNG.
"""
import ast
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_baseline as fb  # noqa: E402

ROOT = fb.ROOT
RES = os.path.join(ROOT, "results", "T09")
OUT = os.path.join(ROOT, "reports", "T09_summary.md")
VARIANTS = ["kl_h0", "kl_h1", "kl_h2", "mse_h2", "infonce"]
LAMS = [0.1, 1.0, 10.0]
CONFIGS = ["base"] + [f"{v}_l{l:g}" for v in VARIANTS for l in LAMS]
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


def lam_of(cfg):
    return 0.0 if cfg == "base" else float(cfg.rsplit("_l", 1)[1])


def fixed_test(npz):
    Z = dict(np.load(npz))
    m = np.where(Z["mode"] == 1)[0]
    m = m[np.argsort(Z["ep_idx"][m], kind="stable")]
    return Z["support"][m], Z["query"][m], Z["query_y"][m].astype(int), Z["logits"][m]


def t07_teacher_fn():
    """diffusion_teacher as defined at tag t07 (for the teacher gate)."""
    src = subprocess.run(["git", "show", "t07:model.py"], capture_output=True, text=True, check=True, cwd=ROOT).stdout
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "diffusion_teacher")
    import scipy.sparse as scipy_sparse
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    ns = {"np": np, "scipy_sparse": scipy_sparse, "gcn_norm": gcn_norm}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "t07_model", "exec"), ns)
    return ns["diffusion_teacher"]


def main():
    R, missing = {}, []
    gates = {"Ga": [], "Gb": [], "Gc": [], "teacher": []}
    for ds in DATASETS:
        H2 = fb.diffusion_features(ds)
        H1 = fb.diffusion_features_hops(ds, 1)
        for k in SHOTS:
            for s in SEEDS:
                ref = None
                for cfg in CONFIGS:
                    d = run_dir(cfg, ds, k, s)
                    if not all(os.path.exists(os.path.join(d, f)) for f in ("run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy")):
                        missing.append((cfg, ds, k, s))
                        continue
                    run = json.load(open(os.path.join(d, "run.json")))
                    Z = np.load(os.path.join(d, "eval_logits.npz"))
                    Fz = np.load(os.path.join(d, "fixed_eval_logits.npz"))
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
                    sup, qry, y, prob = fixed_test(npz)
                    ev = fb.evaluate_npz(npz, H2, alpha=0.5, l_transform="log")
                    R[(cfg, ds, k, s)] = {
                        "bav": run["final"]["best_acc_valid"], "tab": run["final"]["test_acc_at_best_valid"],
                        "teg": (prob.argmax(-1) == y).mean(-1), "F": ev["acc"], "D": ev["acc_D"],
                        "F_edge": ev["iT"] in (0, len(fb.TL_GRID) - 1) or ev["iS"] in (0, len(fb.S_GRID) - 1),
                        "pres_last": run["epochs"][-1].get("pres_loss_mean"),
                        "probe_emb": fb.cosine_probe_acc(np.load(os.path.join(d, "emb_best.npy")), sup, qry, k),
                        "probe_h1": fb.cosine_probe_acc(H1, sup, qry, k) if cfg == "base" else None,
                    }
            print(ds, k, "done", flush=True)
        # teacher gate: kl_h2 (hops = 2) teacher of the current model code vs the t07 definition
        import model as M
        from utils import load_data

        edges, _, X, _, _, _, _, _, _, _ = load_data(ds)
        gates["teacher"].append((ds, float(np.abs(M.diffusion_teacher(X, edges, 2) - t07_teacher_fn()(X, edges)).max()),
                                 float(np.abs(M.diffusion_teacher(X, edges, 2) - H2).max())))
        del H1, H2

    complete = lambda ds, k, cfgs: all((c, ds, k, s) in R for c in cfgs for s in SEEDS)
    sel = {}
    for ds, k in CELLS:
        for v in VARIANTS:
            cands = [f"{v}_l{l:g}" for l in LAMS]
            if complete(ds, k, cands):
                score = {c: np.mean([R[(c, ds, k, s)]["bav"] for s in SEEDS]) for c in cands}
                sel[(v, ds, k)] = sorted(cands, key=lambda c: (-score[c], lam_of(c)))[0]

    L = ["# T09 요약 — 보존 손실 성분 분해 + 자기지도 기준선", ""]
    L.append(f"- run 수: **{len(R)} / 288** (16 설정 × 3 데이터셋 × 5-way {{1,5}}-shot × seed 0–2). 누락: {missing or '없음'}.")
    L.append("- 정확도 %, seed 3개 평균 ± sd. 짝지은 차: 고정 test 에피소드 600개 평균 ± SE (양수 비율, seed 부호 일치 수/3), %p. 풀 `nb`, τ = 0.1, m = 1,024.")
    L.append("- 변형: `kl_hk` = KL 보존, 교사 행 정규화 Â^kX / `mse_h2` = 코사인 유사도 MSE, 교사 Â²X / `infonce` = GRACE식 두 뷰 대조(교사 없음). λ는 (변형, 데이터셋, shot)마다 원본 best_acc_valid seed 평균 최고(동률이면 작은 λ).")
    L.append("")

    L.append("## 표 1 — 전체 격자")
    L.append("| 데이터셋 | shot | 설정 | 원본 best_acc_valid | 원본 test_acc_at_best_valid | 고정 test | F(cfg) 고정 test | 마지막 에폭 pres_loss_mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for c in CONFIGS:
            if not complete(ds, k, [c]):
                continue
            rr = [R[(c, ds, k, s)] for s in SEEDS]
            pl = "—" if lam_of(c) == 0 else mean_sd([r["pres_last"] for r in rr], "{:.4f}")[0]
            L.append(f"| {ds} | {k} | {c} | {mean_sd([100 * r['bav'] for r in rr])[0]} | {mean_sd([100 * r['tab'] for r in rr])[0]} | "
                     f"{mean_sd([100 * r['teg'].mean() for r in rr])[0]} | {mean_sd([100 * r['F'].mean() for r in rr])[0]} | {pl} |")
    L.append("")

    L.append("## 표 2 — 변형별 선택(λ) 결과 (고정 test)")
    L.append("| 데이터셋 | shot | 변형 | 선택 λ | 고정 test | 차 vs base | 차 vs kl_h2(선택) | 차 vs F(base) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        B = [R[("base", ds, k, s)] for s in SEEDS] if complete(ds, k, ["base"]) else None
        if B is None:
            continue
        L.append(f"| {ds} | {k} | base | — | {mean_sd([100 * r['teg'].mean() for r in B])[0]} | — | — | {paired([100 * (b['teg'] - b['F']) for b in B])[0]} |")
        K2 = [R[(sel[('kl_h2', ds, k)], ds, k, s)] for s in SEEDS] if ("kl_h2", ds, k) in sel else None
        for v in VARIANTS:
            if (v, ds, k) not in sel:
                continue
            S = [R[(sel[(v, ds, k)], ds, k, s)] for s in SEEDS]
            vs_k2 = "—" if v == "kl_h2" or K2 is None else paired([100 * (a["teg"] - b["teg"]) for a, b in zip(S, K2)])[0]
            L.append(f"| {ds} | {k} | {v} | {lam_of(sel[(v, ds, k)]):g} | {mean_sd([100 * r['teg'].mean() for r in S])[0]} | "
                     f"{paired([100 * (a['teg'] - b['teg']) for a, b in zip(S, B)])[0]} | {vs_k2} | {paired([100 * (a['teg'] - b['F']) for a, b in zip(S, B)])[0]} |")
    L.append("")

    L.append("## 표 3 — 표현 진단 (선택 모델의 emb_best.npy 코사인 프로토타입 probe, 고정 test, %)")
    L.append("| 데이터셋 | shot | base | " + " | ".join(VARIANTS) + " | Â¹X (행 정규화) |")
    L.append("|---|---|---|" + "---|" * (len(VARIANTS) + 1))
    for ds, k in CELLS:
        if not complete(ds, k, ["base"]):
            continue
        cols = [mean_sd([100 * R[("base", ds, k, s)]["probe_emb"].mean() for s in SEEDS])[0]]
        cols += [mean_sd([100 * R[(sel[(v, ds, k)], ds, k, s)]["probe_emb"].mean() for s in SEEDS])[0] if (v, ds, k) in sel else "—" for v in VARIANTS]
        cols.append(mean_sd([100 * R[("base", ds, k, s)]["probe_h1"].mean() for s in SEEDS])[0])
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")

    pairs = {"L1": ("kl_h2", "infonce"), "L2": ("kl_h2", "kl_h0"), "L3": ("kl_h2", "mse_h2"), "L4": ("infonce", "base")}
    J = {}
    L.append("## 판정 (선택 모델끼리, 평균 > 0 이고 평균 > 2·SE인 (데이터셋, shot) 수)")
    L.append("| 판정 | 비교 | " + " | ".join(f"{ds} {k}" for ds, k in CELLS) + " | 값 |")
    L.append("|---|---|" + "---|" * (len(CELLS) + 1))
    for name, (a, b) in pairs.items():
        cnt, cols = 0, []
        for ds, k in CELLS:
            ca = sel.get((a, ds, k)) if a != "base" else "base"
            cb = sel.get((b, ds, k)) if b != "base" else "base"
            if ca is None or cb is None or not complete(ds, k, [ca, cb]):
                cols.append("—")
                continue
            txt, m, se = paired([100 * (R[(ca, ds, k, s)]["teg"] - R[(cb, ds, k, s)]["teg"]) for s in SEEDS])
            cnt += m > 0 and m > 2 * se
            cols.append(txt)
        J[name] = cnt
        L.append(f"| {name} | {a} − {b} | " + " | ".join(cols) + f" | {cnt}/6 |")
    L.append("")
    edge = [key for key, r in R.items() if r["F_edge"]]
    L.append(f"- F(cfg) 온도 격자 끝 최적값: {len(edge)}개 run {edge[:8] if edge else ''}")
    L.append("")
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    for p, (name, thr) in {"P58": ("L1", 4), "P59": ("L2", 3), "P60": ("L3", 2), "P61": ("L4", 3)}.items():
        L.append(f"| {p} | {name} ≥ {thr}/6 | {name} = {J[name]}/6 | {'일치' if J[name] >= thr else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")

    G = [f"run.json+npz+emb complete: {len(R)}/288; missing {missing or 'none'}",
         f"Ga episodes (eval_logits all epochs + fixed) identical across 16 configs: mismatches {gates['Ga'] or 'none'}",
         f"Gb checkpoint re-evaluation at ckpt_epoch 50/50: failures {gates['Gb'] or 'none'}",
         f"Gc pres_loss_mean finite at every epoch (lambda > 0): failures {gates['Gc'] or 'none'}",
         "teacher (kl_h2, hops=2) max |current - t07 diffusion_teacher|, max |current - fusion_baseline|: " + str(gates["teacher"]),
         f"selected lambda: {sel}", f"J: {J}"]
    open(os.path.join(RES, "gate_checks.txt"), "w").write("\n".join(G) + "\n")
    print("\n".join(G))


if __name__ == "__main__":
    main()
