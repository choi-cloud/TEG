"""T24 analysis -> results/T24/analysis_<ds>.json, results/T24/analysis.json, reports/T24_summary.md, reports/T24_ladder.png.

(copied from tools/analyze_t21.py and modified)
Representations V: S0 = row-normalized A_hat^2 X (fusion_baseline.diffusion_features_hops(ds, 2)); G = results/T24/gpn
emb_best.npy of the run with the probed shot. S1 ep200 / S2 / S3 are read from results/T21/analysis_<ds>.json (not recomputed).
Probe: analyze_t06b.episodes + fewshot_acc as in tools/analyze_t21.py (episodes from random.Random(1000 + seed), drawn once
per group base -> valid -> test per (ds, k, seed), shared by all V; vectors are passed without pre-normalization), plus the
euclidean prototype probe (analyze_t06b.probe_acc(..., metric="euc")) for S0 and G.
Output level (test episodes of the probe): D = argmax fusion_baseline.view_d_scores(S0, sup, qry, k); GPN output = ckpt_best.pt
(eval mode) classified with run_t24.gpn_logp (upstream test() formula: valuator-weighted prototype, euclidean).
Purity: analyze_t14.knn_purity(V, nodes_g, y) (top-10 cosine), V in {S0, G (5-shot run)}.
Split / labels: results/T22/base split.json / labels.npy (= T24 training split).
Usage  python tools/analyze_t24.py run <ds>   -> results/T24/analysis_<ds>.json
       python tools/analyze_t24.py report     -> gates, tables, figure
"""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_t24 as gpn  # noqa: E402  (first: puts tools/gpn ahead of ROOT/layers for the upstream models import)

sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fusion_baseline as fb  # noqa: E402
from analyze_t06b import BATCH, N_QRY, N_WAY, episodes, fewshot_acc, mean_sd, mean_se, probe_acc, sign_agree  # noqa: E402
from analyze_t14 import knn_purity  # noqa: E402

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
GROUPS = ["base", "valid", "test"]
CELLS = [(ds, k) for ds in DATASETS for k in SHOTS]
R21 = os.path.join(ROOT, "results", "T21")
R22 = os.path.join(ROOT, "results", "T22", "base")
R24 = os.path.join(ROOT, "results", "T24")
T21_REPS = ["S1 ep200", "S2", "S3"]
LADDER = ["S0", "G"] + T21_REPS


def gdir(ds, k, s):
    return os.path.join(R24, "gpn", f"{ds}_5w{k}s", f"seed{s}")


def fewshot_acc_euc(V, eps, k):
    """analyze_t06b.fewshot_acc with metric="euc"."""
    accs = []
    for b in range(0, len(eps), BATCH):
        sup = np.stack([e[0] for e in eps[b : b + BATCH]])
        qry = np.stack([e[1] for e in eps[b : b + BATCH]])
        accs.append(probe_acc(V, sup, qry, k, "euc"))
    return 100 * float(np.concatenate(accs).mean())


def d_acc(S0, eps, k):
    """Per-episode accuracy (%) of D: argmax of fusion_baseline.view_d_scores on S0."""
    sup = np.stack([e[0] for e in eps])
    qry = np.stack([e[1] for e in eps])
    y = np.repeat(np.arange(N_WAY), N_QRY)[None, :]
    return 100 * (fb.view_d_scores(S0, sup, qry, k).argmax(-1) == y).mean(-1)


def load_gpn(ds, k, s, features, adj, dev):
    """ckpt_best.pt -> (embeddings, scores) in eval mode; |d| against the saved emb_best.npy / score_best.npy."""
    ck = torch.load(os.path.join(gdir(ds, k, s), "ckpt_best.pt"), map_location="cpu")
    enc = gpn.GPN_Encoder(nfeat=features.shape[1], nhid=gpn.HIDDEN, dropout=gpn.DROPOUT)
    sc = gpn.GPN_Valuator(nfeat=features.shape[1], nhid=gpn.HIDDEN, dropout=gpn.DROPOUT)
    enc.load_state_dict(ck["encoder"])
    sc.load_state_dict(ck["scorer"])
    enc.to(dev).eval()
    sc.to(dev).eval()
    with torch.no_grad():
        emb, scores = enc(features, adj), sc(features, adj)
    d_emb = float(np.abs(emb.cpu().numpy() - np.load(os.path.join(gdir(ds, k, s), "emb_best.npy"))).max())
    d_sc = float(np.abs(scores.view(-1).cpu().numpy() - np.load(os.path.join(gdir(ds, k, s), "score_best.npy"))).max())
    return emb, scores, {"emb_absdiff": d_emb, "score_absdiff": d_sc, "episode": ck["episode"]}


@torch.no_grad()
def gpn_acc(emb, scores, degrees, eps, k):
    """Per-episode accuracy (%) of the GPN output (upstream test() formula) on probe episodes (class-ordered ids)."""
    y = torch.from_numpy(np.repeat(np.arange(N_WAY), N_QRY)).to(emb.device)
    out = []
    for sup, qry in eps:
        logp = gpn.gpn_logp(emb, scores, degrees, sup, qry, N_WAY, k)
        out.append(float((logp.argmax(1) == y).double().mean()))
    return 100 * np.array(out)


def run(ds):
    torch.set_num_threads(16)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    S0 = fb.diffusion_features_hops(ds, 2)
    from utils import load_data

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        _, adj, features, _, degree, _, _, _, _, _ = load_data(ds)
    finally:
        os.chdir(cwd)
    features, adj, degrees = features.to(dev), adj.to(dev), degree.to(dev)
    out = {"acc": {}, "euc": {}, "pur": {}, "ep_test": {}, "ckpt_check": {}, "finite": {}}
    for s in SEEDS:
        for k in SHOTS:
            ref = os.path.join(R22, f"{ds}_5w{k}s", f"seed{s}")
            sp = json.load(open(os.path.join(ref, "split.json")))
            labels = np.load(os.path.join(ref, "labels.npy"))
            groups = {"base": sp["class_list_train"], "valid": sp["class_list_valid"], "test": sp["class_list_test"]}
            by_class = {c: np.where(labels == int(c))[0].tolist() for g in GROUPS for c in groups[g]}
            G = np.load(os.path.join(gdir(ds, k, s), "emb_best.npy"))
            spaces = {"S0": S0, "G": G}
            for n, V in spaces.items():
                out["finite"][f"{n}|{k}|{s}"] = bool(np.isfinite(V).all())
            emb, scores, chk = load_gpn(ds, k, s, features, adj, dev)
            out["ckpt_check"][f"{k}|{s}"] = chk
            rng = random.Random(1000 + s)
            for g in GROUPS:
                eps = episodes(rng, groups[g], by_class, k)
                for n, V in spaces.items():
                    out["acc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc(V, eps, k)
                    out["euc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc_euc(V, eps, k)
                if g == "test":
                    out["ep_test"][f"D|{k}|{s}"] = d_acc(S0, eps, k).tolist()
                    out["ep_test"][f"GPN|{k}|{s}"] = gpn_acc(emb, scores, degrees, eps, k).tolist()
            if k == 5:
                for g in GROUPS:
                    nodes = np.where(np.isin(labels, [int(c) for c in groups[g]]))[0]
                    for n, V in spaces.items():
                        out["pur"][f"{n}|{s}|{g}"] = knn_purity(V, nodes, labels[nodes])
            print(ds, s, k, "done", flush=True)
    json.dump(out, open(os.path.join(R24, f"analysis_{ds}.json"), "w"), indent=1)


def report():
    A = {ds: json.load(open(os.path.join(R24, f"analysis_{ds}.json"))) for ds in DATASETS}
    A21 = {ds: json.load(open(os.path.join(R21, f"analysis_{ds}.json"))) for ds in DATASETS}
    RUN = {(ds, k, s): json.load(open(os.path.join(gdir(ds, k, s), "run.json"))) for ds, k in CELLS for s in SEEDS}
    CUR = {(ds, k, s): json.load(open(os.path.join(gdir(ds, k, s), "curves.json"))) for ds, k in CELLS for s in SEEDS}

    def acc(ds, n, k, s, g):
        return A[ds]["acc"][f"{n}|{k}|{s}|{g}"] if n in ("S0", "G") else A21[ds]["acc"][f"{n}|{k}|{s}|{g}"]

    def euc(ds, n, k, s, g):
        return A[ds]["euc"][f"{n}|{k}|{s}|{g}"]

    def pur(ds, n, s, g):
        return A[ds]["pur"][f"{n}|{s}|{g}"]

    def ep(ds, n, k, s):
        return np.array(A[ds]["ep_test"][f"{n}|{k}|{s}"])

    def m(xs):
        return float(np.mean(xs))

    # ---- gates
    G = {}
    G["Hd"] = {}
    for key, c in CUR.items():
        last100 = float(np.mean(c["train_acc"][-100:]))
        G["Hd"]["|".join(map(str, key))] = {"train_acc_last100": last100, "best_valid": max(c["valid_acc"]), "valid_at10": c["valid_acc"][0],
                                             "ok": last100 > 0.30 and max(c["valid_acc"]) > c["valid_acc"][0]}
    G["He"] = {f"{ds}|{k}|{s}|{g}": abs(A[ds]["acc"][f"S0|{k}|{s}|{g}"] - A21[ds]["acc"][f"S0|{k}|{s}|{g}"])
               for ds, k in CELLS for s in SEEDS for g in GROUPS}
    vals = [x for ds in DATASETS for key in ("acc", "euc", "pur") for x in A[ds][key].values()]
    vals += [x for ds in DATASETS for v in A[ds]["ep_test"].values() for x in v]
    nan_emb = {f"{ds}|{k}|{s}": bool(np.isnan(np.load(os.path.join(gdir(ds, k, s), "emb_best.npy"))).any()) for ds, k in CELLS for s in SEEDS}
    G["Hg"] = {"all_metrics_finite": bool(np.isfinite(vals).all()), "n_metrics": len(vals), "emb_nan_runs": [k for k, v in nan_emb.items() if v],
               "all_spaces_finite": all(all(A[ds]["finite"].values()) for ds in DATASETS)}
    G["ckpt"] = {"emb_absdiff_max": max(v["emb_absdiff"] for ds in DATASETS for v in A[ds]["ckpt_check"].values()),
                 "score_absdiff_max": max(v["score_absdiff"] for ds in DATASETS for v in A[ds]["ckpt_check"].values()),
                 "episode_match": all(v["episode"] == RUN[(ds, int(key.split("|")[0]), int(key.split("|")[1]))]["best_episode"]
                                      for ds in DATASETS for key, v in A[ds]["ckpt_check"].items())}
    G["labels_equal_load_data"] = all(r["labels_equal_load_data"] for r in RUN.values())
    G["pass"] = {"Hd_fail_runs": [k for k, v in G["Hd"].items() if not v["ok"]], "He": max(G["He"].values()) <= 1e-6,
                 "He_max": max(G["He"].values()), "Hg": G["Hg"]["all_metrics_finite"] and G["Hg"]["all_spaces_finite"] and not G["Hg"]["emb_nan_runs"]}

    # ---- judgements (values only)
    J = {}
    J["J1"] = sum(m([ep(ds, "GPN", k, s).mean() for s in SEEDS]) < m([ep(ds, "D", k, s).mean() for s in SEEDS]) for ds, k in CELLS)
    J["J2"] = sum(m([acc(ds, "G", k, s, "test") for s in SEEDS]) < m([acc(ds, "S0", k, s, "test") for s in SEEDS]) for ds, k in CELLS)
    J["J3"] = sum(m([acc(ds, "G", k, s, "base") for s in SEEDS]) > m([acc(ds, "S0", k, s, "base") for s in SEEDS]) for ds, k in CELLS)
    J["J4"] = sum(m([acc(ds, "G", k, s, "valid") for s in SEEDS]) < m([acc(ds, "S0", k, s, "valid") for s in SEEDS]) for ds, k in CELLS)
    J["J5"] = sum(m([pur(ds, "G", s, "test") for s in SEEDS]) < m([pur(ds, "S0", s, "test") for s in SEEDS]) for ds in DATASETS)
    J["J6"] = {f"{ds}|{k}": mean_se([acc(ds, "G", k, s, "test") - acc(ds, "S3", k, s, "test") for s in SEEDS])[0] for ds, k in CELLS}
    J["J6_euc_J2"] = sum(m([euc(ds, "G", k, s, "test") for s in SEEDS]) < m([euc(ds, "S0", k, s, "test") for s in SEEDS]) for ds, k in CELLS)
    P = {"P126": ("J1 ≥ 4/6", f"J1 = {J['J1']}/6"), "P127": ("J2 ≥ 5/6", f"J2 = {J['J2']}/6"), "P128": ("J3 ≥ 5/6", f"J3 = {J['J3']}/6"),
         "P129": ("J4 ≥ 5/6", f"J4 = {J['J4']}/6"), "P130": ("J5 ≥ 2/3", f"J5 = {J['J5']}/3")}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return f"{mean_se(d)[0]} ({sign_agree(d)}/3)"

    L = ["# T24 요약 — GPN 백본 지움 재현", ""]
    L.append("- S0 = 행 정규화 Â²X, G = GPN 공식 인코더(GCN 2층, 16차원) best-valid 출력(`results/T24/gpn/.../emb_best.npy`, probe shot과 "
             "같은 shot의 run). S1 ep200·S2·S3는 `results/T21/analysis_<ds>.json` 값 전재(T21: S1 = Â²X → MLP base-CE, S2 = TEG γ = 0, "
             "S3 = TEG 원본). seed 0–2.")
    L.append("- probe: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid "
             "→ test 순으로 한 번 뽑아 모든 표현이 공유, 사전 정규화 없음). 정확도 %. 분할 = T22 `base` split.json·labels.npy(T09 `base`와 같음).")
    L.append("- 출력 수준: 같은 500 test 에피소드에서 D = `fusion_baseline.view_d_scores(S0)` argmax, GPN 출력 = `ckpt_best.pt`(eval) + 업스트림 "
             "`test()` 수식(valuator 가중 prototype, 유클리드). GPN 원 프로토콜 = best 에피소드의 test task 50개(query 20) 정확도.")
    L.append("")
    L.append("## T1 — 사다리 (seed 평균 ± sd; S1 ep200·S2·S3는 T21 전재)")
    L.append("| 데이터셋 | k | 표현 | acc_base | acc_valid | acc_test |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for n in LADDER:
            L.append(f"| {ds} | {k} | {n} | " + " | ".join(mean_sd([acc(ds, n, k, s, g) for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T2 — 출력 수준 (test, seed 평균 ± sd)")
    L.append("| 데이터셋 | k | D | GPN 출력 | GPN 원 프로토콜 50-task test | 논문 보고값 |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        L.append(f"| {ds} | {k} | {mean_sd([ep(ds, 'D', k, s).mean() for s in SEEDS])[0]} | {mean_sd([ep(ds, 'GPN', k, s).mean() for s in SEEDS])[0]} | "
                 f"{mean_sd([100 * RUN[(ds, k, s)]['test_acc_at_best'] for s in SEEDS])[0]} |  |")
    L.append("")
    L.append("## T3 — 순도 (top-10 코사인 kNN, seed 평균 ± sd; G는 5-shot run)")
    L.append("| 데이터셋 | 표현 | pur_base | pur_valid | pur_test |")
    L.append("|---|---|---|---|---|")
    for ds in DATASETS:
        for n in ("S0", "G"):
            L.append(f"| {ds} | {n} | " + " | ".join(mean_sd([pur(ds, n, s, g) for s in SEEDS], "{:.4f}")[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T4 — 짝지은 차 (%p)")
    L.append("- seed 단위: 평균 ± SE (부호 일치/3). 에피소드 단위 GPN출력 − D: 3 seed × 500 에피소드 = 1500개의 평균 ± SE (양수 비율).")
    L.append("")
    L.append("| 데이터셋 | k | test: G − S0 | valid: G − S0 | base: G − S0 | test: G − S3 | 에피소드: GPN출력 − D |")
    L.append("|---|---|---|---|---|---|---|")
    T4ep = {}
    for ds, k in CELLS:
        t = {n: {g: [acc(ds, n, k, s, g) for s in SEEDS] for g in GROUPS} for n in ("S0", "G", "S3")}
        d = np.concatenate([ep(ds, "GPN", k, s) - ep(ds, "D", k, s) for s in SEEDS])
        T4ep[f"{ds}|{k}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "pos_frac": float((d > 0).mean()), "n": len(d)}
        e = T4ep[f"{ds}|{k}"]
        L.append(f"| {ds} | {k} | {paired(t['G']['test'], t['S0']['test'])} | {paired(t['G']['valid'], t['S0']['valid'])} | "
                 f"{paired(t['G']['base'], t['S0']['base'])} | {paired(t['G']['test'], t['S3']['test'])} | "
                 f"{e['mean']:+.2f} ± {e['se']:.2f} ({e['pos_frac']:.2f}) |")
    L.append("")
    L.append("## T5 — 유클리드 프로토타입 probe (기록, seed 평균 ± sd)")
    L.append("| 데이터셋 | k | 표현 | acc_base | acc_valid | acc_test |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in CELLS:
        for n in ("S0", "G"):
            L.append(f"| {ds} | {k} | {n} | " + " | ".join(mean_sd([euc(ds, n, k, s, g) for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T6 — 학습 곡선 요약 (seed 평균 ± sd; 정확도 %)")
    L.append("| 데이터셋 | k | best 에피소드 (seed 0/1/2) | valid 최고 | valid@10 | 마지막 100 에피소드 학습 정확도 | Hd 미달 |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, k in CELLS:
        hd = [G["Hd"][f"{ds}|{k}|{s}"] for s in SEEDS]
        bad = [str(s) for s, h in zip(SEEDS, hd) if not h["ok"]]
        L.append(f"| {ds} | {k} | {'/'.join(str(RUN[(ds, k, s)]['best_episode']) for s in SEEDS)} | "
                 f"{mean_sd([100 * h['best_valid'] for h in hd])[0]} | {mean_sd([100 * h['valid_at10'] for h in hd])[0]} | "
                 f"{mean_sd([100 * h['train_acc_last100'] for h in hd])[0]} | {('seed ' + ','.join(bad)) if bad else '없음'} |")
    L.append("")
    L.append("## 판정 (값만, seed 평균, 코사인 probe)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| J1 | acc_test(GPN 출력) < acc_test(D) | {J['J1']}/6 |")
    L.append(f"| J2 | acc_test(G) < acc_test(S0) | {J['J2']}/6 |")
    L.append(f"| J3 | acc_base(G) > acc_base(S0) | {J['J3']}/6 |")
    L.append(f"| J4 | acc_valid(G) < acc_valid(S0) | {J['J4']}/6 |")
    L.append(f"| J5 | pur_test(G) < pur_test(S0) | {J['J5']}/3 |")
    L.append("| J6 (기록) | acc_test(G) − acc_test(S3) 짝지은 차 | " + "; ".join(f"{c.replace('|', ' ')}s {v}" for c, v in J["J6"].items()) + " |")
    L.append(f"| J6 (기록) | 유클리드 probe의 J2: acc_test^euc(G) < acc_test^euc(S0) | {J['J6_euc_J2']}/6 |")
    L.append("")
    L.append("## 사전 등록 (조건과 계산값만)")
    L.append("| 예측 | 조건 | 계산값 |")
    L.append("|---|---|---|")
    for p, (c, obs) in P.items():
        L.append(f"| {p} | {c} | {obs} |")
    L.append("")
    open(os.path.join(ROOT, "reports", "T24_summary.md"), "w").write("\n".join(L))

    # ---- figure (tools/analyze_t21.py figure 1 format, G bar added)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["S0", "G", "S1 ep200", "S2", "S3"]
    labels = ["S0\nÂ²X", "G\nGPN", "S1 ep200", "S2\nγ=0", "S3\nTEG"]  # figure text in English (no Hangul font)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        mu = [m([acc(ds, n, 5, s, "test") for s in SEEDS]) for n in names]
        sd = [float(np.std([acc(ds, n, 5, s, "test") for s in SEEDS], ddof=1)) for n in names]
        cols = ["#6b7280", "#d97706"] + ["#2563eb"] * 3
        ax.bar(range(len(names)), mu, yerr=sd, color=cols, width=0.7, capsize=3, error_kw={"elinewidth": 1})
        ax.axhline(mu[0], color="#6b7280", lw=1, ls="--")
        for i, v in enumerate(mu):
            ax.text(i, v + sd[i] + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8)
        lo = min(mu) - 8
        ax.set_ylim(lo, max(mu) + 4)
        ax.set_title(f"{ds} (k = 5)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("acc_test (%), mean ± sd over seeds 0–2")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "T24_ladder.png"), dpi=150)
    json.dump({"gates": G, "J": J, "P": P, "T4_episode": T4ep}, open(os.path.join(R24, "analysis.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(G["pass"]), json.dumps(J, ensure_ascii=False), json.dumps(G["ckpt"]), G["labels_equal_load_data"])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()
