"""T27 analysis (ogbn-arxiv, TEG original vs D / GPN) -> results/T27/analysis_arxiv.json, results/T27/analysis.json,
reports/T27_summary.md, reports/T27_ladder.png.

Representations (probe, as T26): S0 = analyze_t26.s0_arxiv (load_data row-normalized adj, out-neighbors), S0_sym =
analyze_t26.s0_sym, S0_in (record, user addition 2026-10-18) = row-L2-normalized A_in^2 X with A_in = PyG gcn_norm(edge_index,
add_self_loops=True, flow="source_to_target") (the TEG GCNConv operator), P64 / P16 = analyze_t25.pca(S0), TEGe =
results/T27/teg emb_best.npy of the probed shot, G = results/T26/analysis_arxiv.json values (not recomputed).
Probe episodes: analyze_t06b.episodes (500, random.Random(1000 + seed), base -> valid -> test) + fewshot_acc (cosine).
Output level: TEG fixed test episodes (fixed_eval_logits.npz mode 1, 200 / seed): TEG output = argmax of the dumped logits,
D / D_sym / D_in = analyze_t24.d_acc on S0 / S0_sym / S0_in, GPN output = T26 ckpt_best.pt + analyze_t24.gpn_acc (degree 0 -> 1).
Usage  python tools/analyze_t27.py run      -> results/T27/analysis_arxiv.json
       python tools/analyze_t27.py report   -> gates, tables, figure
"""
import json
import math
import os
import pickle
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_t26  # noqa: E402  (first: run_t24 sets the tools/gpn import path)
from analyze_t24 import d_acc, gpn_acc  # noqa: E402
from analyze_t25 import pca  # noqa: E402
from analyze_t26 import l2rows, load_gpn_t26, s0_arxiv, s0_sym  # noqa: E402

sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from analyze_t06b import N_QRY, N_WAY, episodes, fewshot_acc, mean_sd, mean_se, sign_agree  # noqa: E402
from analyze_t14 import knn_purity  # noqa: E402

DS = "ogbn-arxiv"
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
GROUPS = ["base", "valid", "test"]
R26 = os.path.join(ROOT, "results", "T26")
R27 = os.path.join(ROOT, "results", "T27")
LIT_TEG_5W5S = "62.07 ± 1.72"


def tdir(k, s):
    return os.path.join(R27, "teg", f"{DS}_5w{k}s", f"seed{s}")


def a_in(edges, N, dtype=torch.float32):
    """A_in as a torch sparse (N x N) matrix: out[target] += w * x[source], w from PyG gcn_norm (TEG GCNConv operator)."""
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    ei, w = gcn_norm(edges, None, N, add_self_loops=True, flow="source_to_target", dtype=dtype)
    return torch.sparse_coo_tensor(torch.stack([ei[1], ei[0]]), w, (N, N)).coalesce()


def kd2_check(X, edges, Ain):
    """GCNConv(128 -> 128, cached=False, normalize=True, bias=False) with identity lin weight applied once vs A_in X.
    Gate (instruction §7 addendum 2026-10-18): float64 |d|max <= 1e-8; float32 |d|max and relative difference are recorded."""
    from torch_geometric.nn import GCNConv

    def conv_x(Xd, dtype):
        conv = GCNConv(Xd.shape[1], Xd.shape[1], cached=False, normalize=True, bias=False).to(dtype)
        with torch.no_grad():
            conv.lin.weight.copy_(torch.eye(Xd.shape[1], dtype=dtype))
            return conv(Xd, edges)

    ref32 = torch.sparse.mm(Ain, X)
    d32 = float((conv_x(X, torch.float32) - ref32).abs().max())
    X64 = X.double()
    d64 = float((conv_x(X64, torch.float64) - torch.sparse.mm(a_in(edges, X.shape[0], torch.float64), X64)).abs().max())
    ref_max = float(ref32.abs().max())
    return {"f64_absmax": d64, "f32_absmax": d32, "f32_ref_absmax": ref_max, "f32_rel": d32 / ref_max}


def fixed_eps(k, s):
    """mode 1 (test) episodes of fixed_eval_logits.npz: [(support, query)], TEG per-episode accuracy (%), query_y check."""
    Z = np.load(os.path.join(tdir(k, s), "fixed_eval_logits.npz"))
    m = Z["mode"] == 1
    sup, qry, y, lg = Z["support"][m], Z["query"][m], Z["query_y"][m], Z["logits"][m]
    order_ok = bool((y == np.repeat(np.arange(N_WAY), N_QRY)[None, :]).all())
    teg = 100 * (lg.argmax(-1) == y).mean(-1)
    return [(a, b) for a, b in zip(sup, qry)], teg, order_ok, int(m.sum())


def run():
    torch.set_num_threads(16)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edges, adj, X, labels, degree, tr, va, te, _, N = run_t26.load_arxiv()
    Ain = a_in(edges, N)
    kd2 = kd2_check(X, edges, Ain)
    S0 = s0_arxiv(adj, X)
    S0s, _ = s0_sym(edges, X)
    S0in = l2rows(torch.sparse.mm(Ain, torch.sparse.mm(Ain, X)).numpy().astype(np.float32))
    P64, P16, pinfo = pca(S0)
    deg_fix, _ = run_t26.fix_degree(degree)
    e = edges.numpy()
    y = labels.numpy()
    split = pickle.load(open(run_t26.SPLIT_PKL, "rb"))
    groups = {"base": [int(c) for c in tr], "valid": [int(c) for c in va], "test": [int(c) for c in te]}
    by_class = {c: np.where(y == c)[0].tolist() for g in GROUPS for c in groups[g]}
    A26 = json.load(open(os.path.join(R26, "analysis_arxiv.json")))
    facts = {"in_degree0": int((np.bincount(e[1], minlength=N) == 0).sum()), "out_degree0": int((np.bincount(e[0], minlength=N) == 0).sum()),
             "kd2": kd2, "pca": pinfo}
    features, adjd, degrees = X.to(dev), adj.to(dev), deg_fix.to(dev)
    base = {"S0": S0, "S0_in": S0in, "P64": P64, "P16": P16}
    out = {"facts": facts, "acc": {}, "pur": {}, "fixed": {}, "kc": {}, "kd": {}, "finite": {n: bool(np.isfinite(V).all()) for n, V in base.items()}}
    for s in SEEDS:
        for k in SHOTS:
            d = tdir(k, s)
            sp = json.load(open(os.path.join(d, "split.json")))
            out["kc"][f"{k}|{s}"] = {
                "classes_equal": [sorted(sp["class_list_train"]), sorted(sp["class_list_valid"]), sorted(sp["class_list_test"])] == [sorted(x) for x in split],
                "labels_equal": bool(np.array_equal(np.load(os.path.join(d, "labels.npy")), y))}
            T = np.load(os.path.join(d, "emb_best.npy"))
            out["finite"][f"TEGe|{k}|{s}"] = bool(np.isfinite(T).all())
            spaces = base | {"TEGe": T}
            emb, scores, _ = load_gpn_t26(k, s, features, adjd, dev)
            rng = random.Random(1000 + s)
            for g in GROUPS:
                eps = episodes(rng, groups[g], by_class, k)
                for n, V in spaces.items():
                    out["acc"][f"{n}|{k}|{s}|{g}"] = fewshot_acc(V, eps, k)
                if g == "test":  # Kd: GPN output on the T26 probe test episodes
                    g26 = gpn_acc(emb, scores, degrees, eps, k)
                    out["kd"][f"GPN|{k}|{s}"] = float(np.abs(g26 - np.array(A26["ep_test"][f"GPN|{k}|{s}"])).max())
            for n in ("S0", "P16"):
                for g in GROUPS:
                    out["kd"][f"{n}|{k}|{s}|{g}"] = abs(out["acc"][f"{n}|{k}|{s}|{g}"] - A26["acc"][f"{n}|{k}|{s}|{g}"])
            feps, teg, order_ok, n1 = fixed_eps(k, s)
            out["fixed"][f"{k}|{s}"] = {"n": n1, "query_order_ok": order_ok, "TEG": teg.tolist(), "D": d_acc(S0, feps, k).tolist(),
                                        "D_sym": d_acc(S0s, feps, k).tolist(), "D_in": d_acc(S0in, feps, k).tolist(),
                                        "GPN": gpn_acc(emb, scores, degrees, feps, k).tolist()}
            if k == 5:
                for g in GROUPS:
                    nodes = np.where(np.isin(y, groups[g]))[0]
                    for n in ("S0", "P64", "TEGe"):
                        out["pur"][f"{n}|{s}|{g}"] = knn_purity(spaces[n], nodes, y[nodes])
            print(DS, s, k, "done", flush=True)
    json.dump(out, open(os.path.join(R27, "analysis_arxiv.json"), "w"), indent=1)


def run_times(k, s):
    """From console_times.tsv: SPD interval = "Done." line -> the line after its own trailing blank line ("Done.\\n")."""
    rows = []
    for ln in open(os.path.join(tdir(k, s), "console_times.tsv")):
        a, _, b = ln.rstrip("\n").partition("\t")
        rows.append((float(a), b))
    i = next(j for j, (_, t) in enumerate(rows) if t.strip() == "Done.")
    return {"spd_sec": rows[i + 2][0] - rows[i][0], "spd_next_line": rows[i + 2][1][:60], "last_line_sec": rows[-1][0]}


def report():
    A = json.load(open(os.path.join(R27, "analysis_arxiv.json")))
    A26 = json.load(open(os.path.join(R26, "analysis_arxiv.json")))
    RUN = {(k, s): json.load(open(os.path.join(tdir(k, s), "run.json"))) for k in SHOTS for s in SEEDS}
    MEM = {(k, s): json.load(open(os.path.join(tdir(k, s), "gpu_mem.json"))) for k in SHOTS for s in SEEDS}
    RR = {(r["shot"], r["seed"]): r for r in map(json.loads, open(os.path.join(R27, "teg", "_runs.jsonl")))}
    TM = {(k, s): run_times(k, s) for k in SHOTS for s in SEEDS}
    F = A["facts"]

    def acc(n, k, s, g):
        return A26["acc"][f"G|{k}|{s}|{g}"] if n == "G" else A["acc"][f"{n}|{k}|{s}|{g}"]

    def sm(n, k, g):
        return float(np.mean([acc(n, k, s, g) for s in SEEDS]))

    def pur(n, s, g):
        return A["pur"][f"{n}|{s}|{g}"]

    def fx(n, k, s):
        return np.array(A["fixed"][f"{k}|{s}"][n])

    def fxm(n, k):
        return float(np.mean([fx(n, k, s).mean() for s in SEEDS]))

    # ---- gates
    G = {}
    G["Kc"] = A["kc"]
    G["Kd"] = max(A["kd"].values())
    G["Kd2"] = F["kd2"]
    G["Ke"] = {f"{k}|{s}": RUN[(k, s)]["ckpt_recheck"] | {"ckpt_epoch": RUN[(k, s)]["ckpt_epoch"]} for k in SHOTS for s in SEEDS}
    G["Kf"] = {f"{k}|{s}": {"best_acc_valid": RUN[(k, s)]["final"]["best_acc_valid"], "valid_epoch1": RUN[(k, s)]["epochs"][0]["valid_acc"]}
               for k in SHOTS for s in SEEDS}
    vals = [x for key in ("acc", "pur") for x in A[key].values()] + [x for v in A["fixed"].values() for n in ("TEG", "D", "D_sym", "D_in", "GPN") for x in v[n]]
    nan_emb = [f"{k}|{s}" for k in SHOTS for s in SEEDS if np.isnan(np.load(os.path.join(tdir(k, s), "emb_best.npy"))).any()]
    G["Kh"] = {"all_metrics_finite": bool(np.isfinite(vals).all()), "n_metrics": len(vals), "emb_nan_runs": nan_emb,
               "all_spaces_finite": all(A["finite"].values())}
    G["fixed_check"] = {key: {"n": v["n"], "query_order_ok": v["query_order_ok"]} for key, v in A["fixed"].items()}
    G["pass"] = {"Kc": all(v["classes_equal"] and v["labels_equal"] for v in A["kc"].values()), "Kd": G["Kd"] <= 1e-6, "Kd_max": G["Kd"],
                 "Kd2": G["Kd2"]["f64_absmax"] <= 1e-8, "Kd2_f64_max": G["Kd2"]["f64_absmax"],
                 "Ke": all(v and v["epoch"] == v["ckpt_epoch"] and v["n"] == 50 and v["n_equal"] == 50 for v in G["Ke"].values()),
                 "Kf_fail_runs": [key for key, v in G["Kf"].items() if not v["best_acc_valid"] > v["valid_epoch1"]],
                 "Kh": G["Kh"]["all_metrics_finite"] and G["Kh"]["all_spaces_finite"] and not nan_emb,
                 "fixed_eps": all(v["n"] == 200 and v["query_order_ok"] for v in G["fixed_check"].values())}

    # ---- judgements (values only)
    J = {}
    J["M1"] = sum(fxm("TEG", k) < fxm("D", k) for k in SHOTS)
    J["M2"] = sum(sm("TEGe", k, "test") < sm("S0", k, "test") for k in SHOTS)
    J["M3"] = sum(sm("TEGe", k, "base") > sm("S0", k, "base") for k in SHOTS)
    J["M4"] = sum(sm("TEGe", k, "valid") < sm("S0", k, "valid") for k in SHOTS)
    J["M5"] = sum(sm("TEGe", k, "test") < sm("P64", k, "test") for k in SHOTS)
    J["M6_TEG_minus_GPN"] = {str(k): fxm("TEG", k) - fxm("GPN", k) for k in SHOTS}
    J["M6_Dsym_M1"] = sum(fxm("TEG", k) < fxm("D_sym", k) for k in SHOTS)
    J["M6_pur"] = int(np.mean([pur("TEGe", s, "test") for s in SEEDS]) < np.mean([pur("S0", s, "test") for s in SEEDS]))
    J["M6_Din_M1"] = sum(fxm("TEG", k) < fxm("D_in", k) for k in SHOTS)
    J["M6_TEGe_lt_S0in"] = sum(sm("TEGe", k, "test") < sm("S0_in", k, "test") for k in SHOTS)
    P = {"P139": ("M1 = 2/2", f"M1 = {J['M1']}/2"), "P140": ("M2 = 2/2", f"M2 = {J['M2']}/2"), "P141": ("M3 ≥ 1/2", f"M3 = {J['M3']}/2"),
         "P142": ("M4 ≥ 1/2", f"M4 = {J['M4']}/2"), "P143": ("M5 = 2/2", f"M5 = {J['M5']}/2")}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return f"{mean_se(d)[0]} ({sign_agree(d)}/3)"

    def ep_pair(a, b, k):
        d = np.concatenate([fx(a, k, s) - fx(b, k, s) for s in SEEDS])
        return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "pos_frac": float((d > 0).mean()), "n": len(d)}

    L = ["# T27 요약 — ogbn-arxiv에서 TEG 원본 vs D·GPN", ""]
    L.append("- 표현: S0 = 행 L2 정규화(adj²X), adj = `load_data(\"ogbn-arxiv\")` 행 정규화 D⁻¹(A+I)(out-neighbor). S0_in(기록) = 행 L2 정규화(Â_in²X), "
             "Â_in = PyG `gcn_norm(edge_index, add_self_loops=True, flow=\"source_to_target\")`(TEG GCNConv와 같은 연산자, in-neighbor). "
             "P64 / P16 = S0의 PCA(`random_state=18500`). TEGe = TEG 원본 GCN 임베딩(64, `results/T27/teg/.../emb_best.npy`, probe shot과 같은 shot의 run). "
             "G = GPN 임베딩(T26 `analysis_arxiv.json` 전재). seed 0–2.")
    L.append("- ② probe: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → test, T26과 같은 에피소드).")
    L.append("- ① 출력: TEG 고정 test 에피소드(`fixed_eval_logits.npz` mode 1, 200 / seed, `Random(9000 + seed)`). TEG 출력 = 덤프된 logits argmax, "
             "D / D_sym / D_in = S0 / S0_sym / S0_in 위 support 평균 + 코사인, GPN 출력 = T26 `ckpt_best.pt` + 업스트림 `test()` 수식(degree 0 → 1). 정확도 %.")
    L.append("")
    L.append("## T0 — 사실과 실행")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append("| TEG GCN에 넘기는 엣지 | `load_data`의 `edges`(OGB `edge_index` 원 방향) 그대로 (`embedder.py:25–37`, `model.py:354`, `utils.py:155`) |")
    L.append("| TEG GCN 층 | `GCNConv(..., cached=True, normalize=True)`, `flow` 기본값 `source_to_target` (`layers/GCN.py:10`) |")
    L.append("| TEG GCN 집계 방향 | 노드 t가 엣지 (s → t)의 s(edge_index[0])를 모은다 = in-neighbor (PyG `message_passing.py:343`, `:408`; `gcn_conv.py:270–271`) |")
    L.append("| TEG GCN 정규화 | 자기 루프 추가 후 deg = edge_index[1] 기준(in-degree + 1), 가중치 deg^-½[row]·deg^-½[col] (PyG `gcn_conv.py:98–111`) |")
    L.append("| `load_data` adj | A[edge_index[0], edge_index[1]] = 1, 행 정규화 D⁻¹(A+I): 행 i = out-neighbor와 자신의 평균 (`utils.py:159–162`, `:227–234`) |")
    L.append("| 두 연산자의 방향 | 서로 반대(TEG GCN = in-neighbor, load_data adj = out-neighbor) |")
    L.append("| 앵커 SPD | 무방향 `nx.Graph`, 앵커 16개 BFS (`embedder.py:44–79`) |")
    L.append(f"| in-degree(edge_index[1] 기준) 0 노드 수 | {F['in_degree0']:,} |")
    L.append(f"| out-degree(edge_index[0] 기준) 0 노드 수 | {F['out_degree0']:,} |")
    K2 = F["kd2"]
    L.append(f"| Kd2(부기 기준): float64 GCNConv(128→128, 단위 가중치, bias 없음) X vs Â_in X, \\|Δ\\|max | {K2['f64_absmax']:.2e} |")
    L.append(f"| Kd2(기록): float32 \\|Δ\\|max / Â_in X \\|값\\| 최대 / 상대 차이 | {K2['f32_absmax']:.2e} / {K2['f32_ref_absmax']:.2f} / {K2['f32_rel']:.2e} |")
    t0 = TM[(5, 0)]
    L.append(f"| 첫 run(5-shot seed 0, 단독): SPD 구간 / 전체 / GPU 최대 | {t0['spd_sec']:.1f} s / {RR[(5, 0)]['attempts'][-1]['sec']:.1f} s / {MEM[(5, 0)]['max_mib']} MiB |")
    L.append("")
    L.append("| k | seed | 실행 단계 | SPD 구간 (s) | 전체 (s) | `wall_time_sec` (run.json) | GPU 최대 (MiB) |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SEEDS:
        for k in SHOTS:
            L.append(f"| {k} | {s} | {'1(단독)' if (k, s) == (5, 0) else '2(병렬)'} | {TM[(k, s)]['spd_sec']:.1f} | {RR[(k, s)]['attempts'][-1]['sec']:.1f} | "
                     f"{RUN[(k, s)]['wall_time_sec']:.1f} | {MEM[(k, s)]['max_mib']} |")
    L.append("")
    L.append("- SPD 구간 = `console_times.tsv`에서 \"Done.\" 줄부터 다음 출력 줄까지(`embedder.py:61–79`의 그래프 구성 + SPD). GPU 최대 = 1초 간격 `nvidia-smi` memory.used 최대.")
    L.append("")
    L.append("## T1 — ② 표현 (seed 평균 ± sd; G는 T26 전재)")
    L.append("| k | 표현 | acc_base | acc_valid | acc_test |")
    L.append("|---|---|---|---|---|")
    for k in SHOTS:
        for n in ("S0", "S0_in", "P64", "P16", "TEGe", "G"):
            L.append(f"| {k} | {n} | " + " | ".join(mean_sd([acc(n, k, s, g) for s in SEEDS])[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## T2 — ① 출력 (TEG 고정 test 200 에피소드, seed 평균 ± sd)")
    L.append("| k | D | D_sym | D_in | TEG 출력 | GPN 출력 | TEG 원 보고 방식 `test_acc_at_best_valid` |")
    L.append("|---|---|---|---|---|---|---|")
    for k in SHOTS:
        L.append(f"| {k} | " + " | ".join(mean_sd([fx(n, k, s).mean() for s in SEEDS])[0] for n in ("D", "D_sym", "D_in", "TEG", "GPN")) +
                 f" | {mean_sd([100 * RUN[(k, s)]['final']['test_acc_at_best_valid'] for s in SEEDS])[0]} |")
    L.append("")
    L.append(f"- 참고: VISION 논문 표의 TEG(ogbn-arxiv 5-way 5-shot) {LIT_TEG_5W5S}. 클래스 분할·에피소드 프로토콜이 이 표와 다를 수 있다.")
    L.append("")
    L.append("## T3 — 짝지은 차 (%p)")
    L.append("- seed 단위: 평균 ± SE (부호 일치/3), ② probe. 에피소드 단위: ① 고정 test 3 seed × 200 = 600개의 평균 ± SE (양수 비율).")
    L.append("")
    L.append("| k | test: TEGe − S0 | valid: TEGe − S0 | base: TEGe − S0 | test: TEGe − P64 | 에피소드: TEG − D | 에피소드: GPN − D | 에피소드: TEG − GPN | 에피소드: TEG − D_in (기록) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    T3ep = {}
    for k in SHOTS:
        t = {n: {g: [acc(n, k, s, g) for s in SEEDS] for g in GROUPS} for n in ("S0", "TEGe", "P64")}
        cells = []
        for a, b in (("TEG", "D"), ("GPN", "D"), ("TEG", "GPN"), ("TEG", "D_in")):
            e = ep_pair(a, b, k)
            T3ep[f"{k}|{a}-{b}"] = e
            cells.append(f"{e['mean']:+.2f} ± {e['se']:.2f} ({e['pos_frac']:.2f})")
        L.append(f"| {k} | {paired(t['TEGe']['test'], t['S0']['test'])} | {paired(t['TEGe']['valid'], t['S0']['valid'])} | "
                 f"{paired(t['TEGe']['base'], t['S0']['base'])} | {paired(t['TEGe']['test'], t['P64']['test'])} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## T4 — 순도 (top-10 코사인 kNN, seed 평균 ± sd; TEGe는 5-shot run)")
    L.append("| 표현 | pur_base | pur_valid | pur_test |")
    L.append("|---|---|---|---|")
    for n in ("S0", "P64", "TEGe"):
        L.append(f"| {n} | " + " | ".join(mean_sd([pur(n, s, g) for s in SEEDS], "{:.4f}")[0] for g in GROUPS) + " |")
    L.append("")
    L.append("## 판정 (값만, seed 평균)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| M1 | acc_test(TEG 출력) < acc_test(D) (① 출력) | {J['M1']}/2 |")
    L.append(f"| M2 | acc_test(TEGe) < acc_test(S0) | {J['M2']}/2 |")
    L.append(f"| M3 | acc_base(TEGe) > acc_base(S0) | {J['M3']}/2 |")
    L.append(f"| M4 | acc_valid(TEGe) < acc_valid(S0) | {J['M4']}/2 |")
    L.append(f"| M5 | acc_test(TEGe) < acc_test(P64) | {J['M5']}/2 |")
    L.append("| M6 (기록) | acc_test(TEG 출력) − acc_test(GPN 출력), %p | " + "; ".join(f"k = {k}: {v:+.2f}" for k, v in J["M6_TEG_minus_GPN"].items()) + " |")
    L.append(f"| M6 (기록) | D_sym 기준 M1 | {J['M6_Dsym_M1']}/2 |")
    L.append(f"| M6 (기록) | pur_test(TEGe) < pur_test(S0) | {J['M6_pur']}/1 |")
    L.append(f"| M6 (기록) | D_in 기준 M1 | {J['M6_Din_M1']}/2 |")
    L.append(f"| M6 (기록) | acc_test(TEGe) < acc_test(S0_in) | {J['M6_TEGe_lt_S0in']}/2 |")
    L.append("")
    L.append("## 사전 등록 (조건과 계산값만)")
    L.append("| 예측 | 조건 | 계산값 |")
    L.append("|---|---|---|")
    for p, (c, obs) in P.items():
        L.append(f"| {p} | {c} | {obs} |")
    L.append("")
    open(os.path.join(ROOT, "reports", "T27_summary.md"), "w").write("\n".join(L))

    # ---- figure: k = 5, S0 / P64 / TEGe / P16 / G, acc_base and acc_test
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["S0", "P64", "TEGe", "P16", "G"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.38
    for j, (g, col) in enumerate((("base", "#6b7280"), ("test", "#2563eb"))):
        mu = [sm(n, 5, g) for n in names]
        sd = [float(np.std([acc(n, 5, s, g) for s in SEEDS], ddof=1)) for n in names]
        xs = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(xs, mu, w, yerr=sd, color=col, capsize=3, error_kw={"elinewidth": 1}, label=f"acc_{g}")
        for x, v, e in zip(xs, mu, sd):
            ax.text(x, v + e + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["S0\nadj²X", "P64\nPCA64(S0)", "TEGe\nTEG GCN", "P16\nPCA16(S0)", "G\nGPN"], fontsize=8)  # English (no Hangul font)
    allv = [sm(n, 5, g) for n in names for g in ("base", "test")]
    ax.set_ylim(min(allv) - 8, max(allv) + 5)
    ax.set_title("ogbn-arxiv (k = 5)", fontsize=10)
    ax.set_ylabel("accuracy (%), mean ± sd over seeds 0–2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "T27_ladder.png"), dpi=150)
    json.dump({"gates": G, "J": J, "P": P, "T3_episode": T3ep, "times": {f"{k}|{s}": v for (k, s), v in TM.items()}},
              open(os.path.join(R27, "analysis.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(G["pass"]), json.dumps(J))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    else:
        report()
