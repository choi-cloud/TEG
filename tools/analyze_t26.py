"""T26 analysis (ogbn-arxiv) -> results/T26/analysis_arxiv.json, results/T26/analysis.json, reports/T26_summary.md,
reports/T26_ladder.png.

(copied from tools/analyze_t24.py and modified)
Representations: S0 = row-L2-normalized adj^2 X with adj = utils.load_data("ogbn-arxiv") adj (row-normalized D^-1(A+I),
original edge direction; the operator GPN receives). S0_sym (record) = row-L2-normalized (D^-1/2 (A_s+I) D^-1/2)^2 X with
A_s = A ∪ A^T (duplicates removed), computed here (fusion_baseline is not used). P16 (record) = analyze_t25.pca(S0) first 16
(random_state 18500). G = results/T26/gpn emb_best.npy of the run with the probed shot.
Probe: analyze_t06b.episodes + fewshot_acc (cosine prototype, 500 episodes, random.Random(1000 + seed), base -> valid -> test).
Output level (test episodes): D = analyze_t24.d_acc(S0), D_sym = analyze_t24.d_acc(S0_sym), GPN output = ckpt_best.pt (eval)
+ run_t24.gpn_logp with the training degree (degree 0 -> 1). Purity: analyze_t14.knn_purity, S0 and G (5-shot run).
Usage  python tools/analyze_t26.py run      -> results/T26/analysis_arxiv.json
       python tools/analyze_t26.py report   -> gates, tables, figure
"""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_t26  # noqa: E402  (first: run_t24 sets the tools/gpn import path)
from run_t24 import DROPOUT, HIDDEN, GPN_Encoder, GPN_Valuator  # noqa: E402
from analyze_t24 import d_acc, gpn_acc  # noqa: E402
from analyze_t25 import pca  # noqa: E402

sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from analyze_t06b import episodes, fewshot_acc, mean_sd, mean_se, sign_agree  # noqa: E402
from analyze_t14 import knn_purity  # noqa: E402

DS = "ogbn-arxiv"
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
GROUPS = ["base", "valid", "test"]
R26 = os.path.join(ROOT, "results", "T26")
LIT_5W5S = "53.06 ± 3.13"


def gdir(k, s):
    return os.path.join(R26, "gpn", f"{DS}_5w{k}s", f"seed{s}")


def l2rows(D):
    return D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)


def s0_arxiv(adj, X):
    """Row-L2-normalized adj (adj X), adj = load_data row-normalized operator, float32."""
    D = torch.sparse.mm(adj, torch.sparse.mm(adj, X))
    return l2rows(D.numpy().astype(np.float32))


def s0_sym(edges, X):
    """A_s = A ∪ A^T (dedup), D^-1/2 (A_s + I) D^-1/2 two hops, row L2 normalization. Also returns facts for T0."""
    e = edges.numpy()
    N = X.shape[0]
    A = sp.coo_matrix((np.ones(2 * e.shape[1]), (np.r_[e[0], e[1]], np.r_[e[1], e[0]])), shape=(N, N)).tocsr()
    A.data[:] = 1.0  # duplicates summed by tocsr -> set back to 1
    deg_s = np.asarray(A.sum(1)).ravel()
    facts = {"sym_nnz_directed": int(A.nnz), "sym_undirected_edges": int(sp.triu(A, k=1).nnz), "sym_self_loops": int(A.diagonal().sum()),
             "sym_degree0": int((deg_s == 0).sum())}
    Ah = A + sp.eye(N, format="csr")
    d = np.asarray(Ah.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d))
    Ah = (dinv @ Ah @ dinv).astype(np.float32).tocsr()
    Xn = X.numpy().astype(np.float32)
    D = Ah @ (Ah @ Xn)
    return l2rows(D.astype(np.float32)), facts


def load_gpn_t26(k, s, features, adj, dev):
    """analyze_t24.load_gpn with the T26 path."""
    ck = torch.load(os.path.join(gdir(k, s), "ckpt_best.pt"), map_location="cpu")
    enc = GPN_Encoder(nfeat=features.shape[1], nhid=HIDDEN, dropout=DROPOUT)
    sc = GPN_Valuator(nfeat=features.shape[1], nhid=HIDDEN, dropout=DROPOUT)
    enc.load_state_dict(ck["encoder"])
    sc.load_state_dict(ck["scorer"])
    enc.to(dev).eval()
    sc.to(dev).eval()
    with torch.no_grad():
        emb, scores = enc(features, adj), sc(features, adj)
    d_emb = float(np.abs(emb.cpu().numpy() - np.load(os.path.join(gdir(k, s), "emb_best.npy"))).max())
    d_sc = float(np.abs(scores.view(-1).cpu().numpy() - np.load(os.path.join(gdir(k, s), "score_best.npy"))).max())
    return emb, scores, {"emb_absdiff": d_emb, "score_absdiff": d_sc, "episode": ck["episode"]}


def run():
    torch.set_num_threads(16)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edges, adj, X, labels, degree, tr, va, te, _, N = run_t26.load_arxiv()
    S0 = s0_arxiv(adj, X)
    S0s, facts = s0_sym(edges, X)
    P64, P16, pinfo = pca(S0)
    deg_fix, n0 = run_t26.fix_degree(degree)
    e = edges.numpy()
    key, rkey = e[0].astype(np.int64) * N + e[1], e[1].astype(np.int64) * N + e[0]
    y = labels.numpy()
    groups = {"base": [int(c) for c in tr], "valid": [int(c) for c in va], "test": [int(c) for c in te]}
    by_class = {c: np.where(y == c)[0].tolist() for g in GROUPS for c in groups[g]}
    facts |= {"nodes": int(N), "edges": int(e.shape[1]), "feat_dim": int(X.shape[1]), "self_loops": int((e[0] == e[1]).sum()),
              "reverse_edge_ratio": float(np.isin(rkey, key).mean()), "n_class": {g: len(groups[g]) for g in GROUPS},
              "min_class_size": {g: int(min((y == c).sum() for c in groups[g])) for g in GROUPS},
              "nodes_per_group": {g: int(np.isin(y, groups[g]).sum()) for g in GROUPS},
              "degree0": n0, "degree1": int((degree.numpy().ravel() == 1).sum()),
              "out_degree0_rows_equal_X": bool(np.allclose(S0[(degree.numpy().ravel() == 0)], l2rows(X.numpy()[(degree.numpy().ravel() == 0)]), atol=1e-5))}
    features, adjd, degrees = X.to(dev), adj.to(dev), deg_fix.to(dev)
    spaces0 = {"S0": S0, "S0_sym": S0s, "P16": P16}
    out = {"facts": facts, "pca": pinfo, "acc": {}, "pur": {}, "ep_test": {}, "ckpt_check": {}, "finite": {}}
    for n, V in spaces0.items():
        out["finite"][n] = bool(np.isfinite(V).all())
    for s in SEEDS:
        for k in SHOTS:
            G = np.load(os.path.join(gdir(k, s), "emb_best.npy"))
            spaces = spaces0 | {"G": G}
            out["finite"][f"G|{k}|{s}"] = bool(np.isfinite(G).all())
            emb, scores, chk = load_gpn_t26(k, s, features, adjd, dev)
            out["ckpt_check"][f"{k}|{s}"] = chk
            rng = random.Random(1000 + s)
            for g in GROUPS:
                eps = episodes(rng, groups[g], by_class, k)
                for n, V in spaces.items():
                    out["acc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc(V, eps, k)
                if g == "test":
                    out["ep_test"][f"D|{k}|{s}"] = d_acc(S0, eps, k).tolist()
                    out["ep_test"][f"D_sym|{k}|{s}"] = d_acc(S0s, eps, k).tolist()
                    out["ep_test"][f"GPN|{k}|{s}"] = gpn_acc(emb, scores, degrees, eps, k).tolist()
            if k == 5:
                for g in GROUPS:
                    nodes = np.where(np.isin(y, groups[g]))[0]
                    for n in ("S0", "G"):
                        out["pur"][f"{n}|{s}|{g}"] = knn_purity(spaces[n], nodes, y[nodes])
            print(DS, s, k, "done", flush=True)
    json.dump(out, open(os.path.join(R26, "analysis_arxiv.json"), "w"), indent=1)


def report():
    A = json.load(open(os.path.join(R26, "analysis_arxiv.json")))
    RUN = {(k, s): json.load(open(os.path.join(gdir(k, s), "run.json"))) for k in SHOTS for s in SEEDS}
    CUR = {(k, s): json.load(open(os.path.join(gdir(k, s), "curves.json"))) for k in SHOTS for s in SEEDS}
    hcj = json.load(open(os.path.join(R26, "hc_check.json")))
    F = A["facts"]

    def acc(n, k, s, g):
        return A["acc"][f"{n}|{k}|{s}|{g}"]

    def sm(n, k, g):
        return float(np.mean([acc(n, k, s, g) for s in SEEDS]))

    def pur(n, s, g):
        return A["pur"][f"{n}|{s}|{g}"]

    def ep(n, k, s):
        return np.array(A["ep_test"][f"{n}|{k}|{s}"])

    def epm(n, k):
        return float(np.mean([ep(n, k, s).mean() for s in SEEDS]))

    # ---- gates
    G = {"Jb": hcj["max"], "Jb_pass": hcj["pass"] and all(v == 0.0 for v in hcj["max"].values())}
    G["Jc"] = {}
    for (k, s), c in CUR.items():
        last100 = float(np.mean(c["train_acc"][-100:]))
        G["Jc"][f"{k}|{s}"] = {"train_acc_last100": last100, "best_valid": max(c["valid_acc"]), "valid_at10": c["valid_acc"][0],
                               "ok": last100 > 0.30 and max(c["valid_acc"]) > c["valid_acc"][0]}
    vals = [x for key in ("acc", "pur") for x in A[key].values()] + [x for v in A["ep_test"].values() for x in v]
    nan_emb = [f"{k}|{s}" for k in SHOTS for s in SEEDS if np.isnan(np.load(os.path.join(gdir(k, s), "emb_best.npy"))).any()]
    G["Jf"] = {"all_metrics_finite": bool(np.isfinite(vals).all()), "n_metrics": len(vals), "emb_nan_runs": nan_emb,
               "all_spaces_finite": all(A["finite"].values())}
    G["ckpt"] = {"emb_absdiff_max": max(v["emb_absdiff"] for v in A["ckpt_check"].values()),
                 "score_absdiff_max": max(v["score_absdiff"] for v in A["ckpt_check"].values()),
                 "episode_match": all(v["episode"] == RUN[tuple(map(int, key.split("|")))]["best_episode"] for key, v in A["ckpt_check"].items())}
    G["pass"] = {"Jb": G["Jb_pass"], "Jc_fail_runs": [k for k, v in G["Jc"].items() if not v["ok"]],
                 "Jf": G["Jf"]["all_metrics_finite"] and G["Jf"]["all_spaces_finite"] and not nan_emb}

    # ---- judgements (values only)
    J = {}
    J["L1"] = sum(epm("GPN", k) < epm("D", k) for k in SHOTS)
    J["L2"] = sum(sm("G", k, "test") < sm("S0", k, "test") for k in SHOTS)
    J["L3"] = sum(sm("G", k, "base") > sm("S0", k, "base") for k in SHOTS)
    J["L4"] = sum(sm("G", k, "valid") < sm("S0", k, "valid") for k in SHOTS)
    J["L5_P16"] = sum(sm("G", k, "test") < sm("P16", k, "test") for k in SHOTS)
    J["L5_pur"] = int(np.mean([pur("G", s, "test") for s in SEEDS]) < np.mean([pur("S0", s, "test") for s in SEEDS]))
    J["L5_Dsym_L1"] = sum(epm("GPN", k) < epm("D_sym", k) for k in SHOTS)
    P = {"P135": ("L3 = 2/2", f"L3 = {J['L3']}/2"), "P136": ("L2 ≥ 1/2", f"L2 = {J['L2']}/2"),
         "P137": ("L4 ≥ 1/2", f"L4 = {J['L4']}/2"), "P138": ("L1 ≤ 1/2", f"L1 = {J['L1']}/2")}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return f"{mean_se(d)[0]} ({sign_agree(d)}/3)"

    L = ["# T26 요약 — ogbn-arxiv(입력 128차원)에서 D vs GPN", ""]
    L.append("- S0 = 행 L2 정규화(adj²X), adj = `utils.load_data(\"ogbn-arxiv\")`의 행 정규화 D⁻¹(A+I)(원 엣지 방향, GPN이 받는 연산자). "
             "S0_sym(기록) = A ∪ Aᵀ(중복 제거) + I의 대칭 정규화 D^-1/2 (A+I) D^-1/2로 2-hop 전파 후 행 L2 정규화(`analyze_t26.s0_sym`). "
             "P16(기록) = S0의 PCA 16차원(`random_state=18500`). G = GPN best-valid 임베딩(16차원, probe shot과 같은 shot의 run). seed 0–2.")
    L.append("- probe: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → "
             "test 순). 클래스 분할 = `dataset/ogbn_arxiv/cls_split.pkl`(seed와 무관). 정확도 %.")
    L.append("- 출력 수준: D = S0 위 support 평균 + 코사인, D_sym = S0_sym 위 같은 규칙, GPN 출력 = `ckpt_best.pt`(eval) + 업스트림 `test()` 수식"
             "(degree 0 노드는 학습과 같이 degree 1).")
    L.append("")
    L.append("## T0 — 사전 확인 사실")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    rows = [("노드 수", f"{F['nodes']:,}"), ("엣지 수(`edge_index`)", f"{F['edges']:,}"), ("특징 차원", F["feat_dim"]),
            ("self-loop", F["self_loops"]), ("역방향 엣지가 있는 비율", f"{F['reverse_edge_ratio']:.4f}"),
            ("클래스 수 train / valid / test", " / ".join(str(F["n_class"][g]) for g in GROUPS)),
            ("그룹별 노드 수 train / valid / test", " / ".join(f"{F['nodes_per_group'][g]:,}" for g in GROUPS)),
            ("그룹별 최소 클래스 크기 train / valid / test", " / ".join(str(F["min_class_size"][g]) for g in GROUPS)),
            ("`degree`(load_data, A 행합) = 0 노드 수 (GPN에서 1로 둠)", f"{F['degree0']:,}"), ("`degree` = 1 노드 수", f"{F['degree1']:,}"),
            ("대칭화 A ∪ Aᵀ 엣지 수 (비대칭 항목 수 / 무방향 쌍 수)", f"{F['sym_nnz_directed']:,} / {F['sym_undirected_edges']:,}"),
            ("대칭화 후 degree 0 노드 수", f"{F['sym_degree0']:,}"),
            ("out-degree 0 노드의 S0 행 = 행 정규화 X", "예" if F["out_degree0_rows_equal_X"] else "아니오"),
            ("PCA(S0) 누적 설명 분산 16 / 64", f"{A['pca']['cum16']:.4f} / {A['pca']['cum64']:.4f}")]
    for a, b in rows:
        L.append(f"| {a} | {b} |")
    L.append("")
    L.append("## T1 — (seed 평균 ± sd)")
    L.append("| k | 표현 | acc_base | acc_valid | acc_test |")
    L.append("|---|---|---|---|---|")
    for k in SHOTS:
        for n in ("S0", "S0_sym", "P16", "G"):
            L.append(f"| {k} | {n} | " + " | ".join(mean_sd([acc(n, k, s, g) for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T2 — 출력 수준 (test, seed 평균 ± sd)")
    L.append("| k | D | D_sym | GPN 출력 | GPN 원 프로토콜 50-task test | 참고: 문헌값 |")
    L.append("|---|---|---|---|---|---|")
    for k in SHOTS:
        L.append(f"| {k} | {mean_sd([ep('D', k, s).mean() for s in SEEDS])[0]} | {mean_sd([ep('D_sym', k, s).mean() for s in SEEDS])[0]} | "
                 f"{mean_sd([ep('GPN', k, s).mean() for s in SEEDS])[0]} | {mean_sd([100 * RUN[(k, s)]['test_acc_at_best'] for s in SEEDS])[0]} | "
                 f"{LIT_5W5S + ' (TENT 논문 표의 GPN, 5w5s)' if k == 5 else ''} |")
    L.append("")
    L.append("- 문헌값은 TENT 논문 표의 GPN(ogbn-arxiv 5-way 5-shot)이다. 클래스 분할·에피소드 프로토콜이 이 표와 다를 수 있다.")
    L.append("")
    L.append("## T3 — 짝지은 차 (%p)")
    L.append("- seed 단위: 평균 ± SE (부호 일치/3). 에피소드 단위: 3 seed × 500 에피소드 = 1500개의 평균 ± SE (양수 비율).")
    L.append("")
    L.append("| k | test: G − S0 | valid: G − S0 | base: G − S0 | test: G − P16 | 에피소드: GPN출력 − D | 에피소드: GPN출력 − D_sym (기록) |")
    L.append("|---|---|---|---|---|---|---|")
    T3ep = {}
    for k in SHOTS:
        cells = []
        for ref in ("D", "D_sym"):
            d = np.concatenate([ep("GPN", k, s) - ep(ref, k, s) for s in SEEDS])
            T3ep[f"{k}|{ref}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "pos_frac": float((d > 0).mean()), "n": len(d)}
            cells.append(f"{d.mean():+.2f} ± {d.std(ddof=1) / math.sqrt(len(d)):.2f} ({(d > 0).mean():.2f})")
        t = {n: {g: [acc(n, k, s, g) for s in SEEDS] for g in GROUPS} for n in ("S0", "G", "P16")}
        L.append(f"| {k} | {paired(t['G']['test'], t['S0']['test'])} | {paired(t['G']['valid'], t['S0']['valid'])} | "
                 f"{paired(t['G']['base'], t['S0']['base'])} | {paired(t['G']['test'], t['P16']['test'])} | {cells[0]} | {cells[1]} |")
    L.append("")
    L.append("## T4 — 순도 (top-10 코사인 kNN, seed 평균 ± sd; G는 5-shot run)")
    L.append("| 표현 | pur_base | pur_valid | pur_test |")
    L.append("|---|---|---|---|")
    for n in ("S0", "G"):
        L.append(f"| {n} | " + " | ".join(mean_sd([pur(n, s, g) for s in SEEDS], "{:.4f}")[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T5 — 학습 곡선 요약, 시간·GPU 메모리 (정확도 %)")
    L.append("| k | seed | best 에피소드 | valid 최고 | valid@10 | 마지막 100 에피소드 학습 정확도 | 학습 시간 (s) | GPU 최대 할당 / 예약 (MB) | Jc |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in SHOTS:
        for s in SEEDS:
            j, r = G["Jc"][f"{k}|{s}"], RUN[(k, s)]
            L.append(f"| {k} | {s} | {r['best_episode']} | {100 * j['best_valid']:.2f} | {100 * j['valid_at10']:.2f} | {100 * j['train_acc_last100']:.2f} | "
                     f"{r['train_sec']:.0f} | {r['gpu_max_allocated_mb']:.0f} / {r['gpu_max_reserved_mb']:.0f} | {'충족' if j['ok'] else '미달'} |")
    L.append("")
    L.append("## 판정 (값만, seed 평균, 코사인 probe)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| L1 | acc_test(GPN 출력) < acc_test(D) | {J['L1']}/2 |")
    L.append(f"| L2 | acc_test(G) < acc_test(S0) | {J['L2']}/2 |")
    L.append(f"| L3 | acc_base(G) > acc_base(S0) | {J['L3']}/2 |")
    L.append(f"| L4 | acc_valid(G) < acc_valid(S0) | {J['L4']}/2 |")
    L.append(f"| L5 (기록) | acc_test(G) < acc_test(P16); pur_test(G) < pur_test(S0); D_sym 기준 L1 | {J['L5_P16']}/2; {J['L5_pur']}/1; {J['L5_Dsym_L1']}/2 |")
    L.append("")
    L.append("## 사전 등록 (조건과 계산값만)")
    L.append("| 예측 | 조건 | 계산값 |")
    L.append("|---|---|---|")
    for p, (c, obs) in P.items():
        L.append(f"| {p} | {c} | {obs} |")
    L.append("")
    open(os.path.join(ROOT, "reports", "T26_summary.md"), "w").write("\n".join(L))

    # ---- figure: k = 5, S0 / P16 / G, acc_base and acc_test
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["S0", "P16", "G"]
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.38
    for j, (g, col) in enumerate((("base", "#6b7280"), ("test", "#2563eb"))):
        mu = [sm(n, 5, g) for n in names]
        sd = [float(np.std([acc(n, 5, s, g) for s in SEEDS], ddof=1)) for n in names]
        xs = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(xs, mu, w, yerr=sd, color=col, capsize=3, error_kw={"elinewidth": 1}, label=f"acc_{g}")
        for x, v, e in zip(xs, mu, sd):
            ax.text(x, v + e + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["S0\nadj²X", "P16\nPCA16(S0)", "G\nGPN"], fontsize=8)  # figure text in English (no Hangul font)
    allv = [sm(n, 5, g) for n in names for g in ("base", "test")]
    ax.set_ylim(min(allv) - 8, max(allv) + 5)
    ax.set_title("ogbn-arxiv (k = 5)", fontsize=10)
    ax.set_ylabel("accuracy (%), mean ± sd over seeds 0–2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "T26_ladder.png"), dpi=150)
    json.dump({"gates": G, "J": J, "P": P, "T3_episode": T3ep}, open(os.path.join(R26, "analysis.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(G["pass"]), json.dumps(J), json.dumps(G["ckpt"]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    else:
        report()
