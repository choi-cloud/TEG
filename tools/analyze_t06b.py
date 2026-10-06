"""T06b analysis -> reports/T06b_summary.md.

Inputs: results/T06b/L{1,2}/<ds>_5w<k>s/seed<s>/{run.json, emb_best.npy, split.json, labels.npy, test_eps.npz, episodes.jsonl},
results/T06 (L1 reference accuracies), and the datasets via utils.load_data (raw / diffusion / control spaces).
Labels are used for analysis only. Analysis RNG (T06b §3): probe episodes random.Random(1000+seed),
random projection np.random.default_rng(2000+seed), base split random.Random(3000+seed),
permutations np.random.default_rng(4000+seed) (a fresh instance per (row, seed)). PCA: randomized SVD, random_state=0.
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from utils import load_data  # noqa: E402

R6B = os.path.join(ROOT, "results", "T06b")
R06 = os.path.join(ROOT, "results", "T06")
OUT = os.path.join(ROOT, "reports", "T06b_summary.md")

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
LAYERS = [1, 2]
GROUPS = ["base", "valid", "test"]
SPACES = ["raw", "diff1", "diff2", "rand1", "rand2", "pca1", "pca2", "emb1", "emb2"]
N_WAY, N_QRY, N_EPI = 5, 5, 500
N_SPLIT, N_PERM = 20, 1000
VARIANTS = [("0.80", 0.80), ("0.90", 0.90), ("0.95", 0.95), ("r=10", 10)]
AXIS_SPACES = ["diff1", "diff2", "pca1", "pca2"]
BATCH = 50


# ---------------------------------------------------------------- helpers
def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def mean_se(xs, fmt="{:+.2f}"):
    xs = np.asarray(xs, dtype=float)
    m, se = xs.mean(), xs.std(ddof=1) / math.sqrt(len(xs))
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(se)}", m


def sign_agree(xs):
    m = sum(xs) / len(xs)
    sg = (m > 0) - (m < 0)
    return sum(1 for x in xs if sg != 0 and ((x > 0) - (x < 0)) == sg)


def l2n(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def run_dir(layer, ds, k, s):
    return os.path.join(R6B, f"L{layer}", f"{ds}_5w{k}s", f"seed{s}")


# ---------------------------------------------------------------- spaces
def dataset_spaces(ds):
    edges, _, X, _, _, _, _, _, _, _ = load_data(ds)
    X = X.numpy().astype(np.float32)
    ei, w = gcn_norm(edges, None, X.shape[0], add_self_loops=True)
    ei, w = ei.numpy(), w.numpy()
    A = sp.csr_matrix((w, (ei[1], ei[0])), shape=(X.shape[0], X.shape[0]))  # out[target] += w * x[source]
    D1 = (A @ X).astype(np.float32)
    D2 = (A @ D1).astype(np.float32)
    P1 = PCA(n_components=64, svd_solver="randomized", random_state=0).fit_transform(D1).astype(np.float32)
    P2 = PCA(n_components=64, svd_solver="randomized", random_state=0).fit_transform(D2).astype(np.float32)
    return {"raw": X, "diff1": D1, "diff2": D2, "pca1": P1, "pca2": P2}


def rand_spaces(base, seed):
    R = np.random.default_rng(2000 + seed).normal(0.0, 1.0 / 8.0, size=(base["diff1"].shape[1], 64)).astype(np.float32)  # N(0, 1/64)
    return {"rand1": base["diff1"] @ R, "rand2": base["diff2"] @ R}


# ---------------------------------------------------------------- check 1 (probe)
def episodes(rng, classes, by_class, k):
    out = []
    for _ in range(N_EPI):
        cls = rng.sample(classes, N_WAY)
        sup, qry = [], []
        for c in cls:
            ids = rng.sample(by_class[c], k + N_QRY)
            sup.append(ids[:k])
            qry.append(ids[k:])
        out.append((np.array(sup).reshape(-1), np.array(qry).reshape(-1)))
    return out


def probe_acc(V, sup, qry, k, metric="cos"):
    """sup: (B, N*K), qry: (B, N*M) global ids in class order. Returns per-episode accuracy (B,)."""
    B = sup.shape[0]
    P = V[sup].reshape(B, N_WAY, k, V.shape[1]).mean(2)
    Q = V[qry].reshape(B, -1, V.shape[1])
    if metric == "cos":
        score = np.einsum("bqd,bnd->bqn", l2n(Q), l2n(P))
    else:
        score = -(np.sum(Q**2, 2)[:, :, None] - 2 * np.einsum("bqd,bnd->bqn", Q, P) + np.sum(P**2, 2)[:, None, :])
    true = np.repeat(np.arange(N_WAY), qry.shape[1] // N_WAY)[None, :]
    return (score.argmax(2) == true).mean(1)


def fewshot_acc(V, eps, k):
    accs = []
    for b in range(0, len(eps), BATCH):
        sup = np.stack([e[0] for e in eps[b : b + BATCH]])
        qry = np.stack([e[1] for e in eps[b : b + BATCH]])
        accs.append(probe_acc(V, sup, qry, k, "cos"))
    return 100 * float(np.concatenate(accs).mean())


# ---------------------------------------------------------------- check 2 (axis loss)
def basis(U, rule):
    """Principal axes of the ordered-pair differences of the rows of U (equivalently of centered U)."""
    _, s, Vt = np.linalg.svd(U - U.mean(0, keepdims=True), full_matrices=False)
    if isinstance(rule, int):
        r = rule
    else:
        ratio = np.cumsum(s**2) / np.sum(s**2)
        r = int(np.searchsorted(ratio, rule - 1e-12) + 1)
    return Vt[:r]


def resid_matrix(U, Bs):
    """M_ij = ||d - P d||^2 / ||d||^2 for d = u_i - u_j (unit-norm rows), zero diagonal."""
    G = U @ U.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    C = U @ Bs.T
    Gc = C @ C.T
    pq = np.diag(Gc)[:, None] + np.diag(Gc)[None, :] - 2 * Gc
    M = np.zeros_like(sq)
    off = ~np.eye(len(U), dtype=bool)
    M[off] = 1.0 - pq[off] / sq[off]
    return M


def weighted_resid(U, Bs):
    """Variance-weighted residual over ordered pairs: 1 - sum ||P d||^2 / sum ||d||^2 (gate G3, user decision 2026-10-06)."""
    G = U @ U.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    C = U @ Bs.T
    Gc = C @ C.T
    pq = np.diag(Gc)[:, None] + np.diag(Gc)[None, :] - 2 * Gc
    off = ~np.eye(len(U), dtype=bool)
    return float(1.0 - pq[off].sum() / sq[off].sum())


def pair_mean(M):
    n = len(M)
    return float(M[np.triu_indices(n, 1)].mean())


def axis_rows(protos, split, splits, seed):
    """One (space, seed) block. Returns {variant: list of per-split dicts}, principal-angle values."""
    base, valid, test = split["class_list_train"], split["class_list_valid"], split["class_list_test"]
    T = np.stack([protos[c] for c in test])
    Vv = np.stack([protos[c] for c in valid])
    PT = basis(T, 0.90)
    prng = np.random.default_rng(4000 + seed)
    out = defaultdict(list)
    angles = {"A_B": [], "A_T": []}
    for A_cls, B_cls in splits:
        UA = np.stack([protos[c] for c in A_cls])
        UB = np.stack([protos[c] for c in B_cls])
        pool = np.concatenate([UB, T])
        nb, nt = len(UB), len(T)
        keys = prng.random((N_PERM, nb + nt))
        Z = np.zeros((N_PERM, nb + nt))
        np.put_along_axis(Z, np.argsort(keys, 1)[:, :nt], 1.0, 1)
        for name, rule in VARIANTS:
            BA = basis(UA, rule)
            rA, rB = pair_mean(resid_matrix(UA, BA)), pair_mean(resid_matrix(UB, BA))
            rT, rV = pair_mean(resid_matrix(T, BA)), pair_mean(resid_matrix(Vv, BA))
            Mp = resid_matrix(pool, BA)
            g = rT - rB
            sumT = np.einsum("pi,ij,pj->p", Z, Mp, Z) / 2
            sumB = np.einsum("pi,ij,pj->p", 1 - Z, Mp, 1 - Z) / 2
            gperm = sumT / (nt * (nt - 1) / 2) - sumB / (nb * (nb - 1) / 2)
            out[name].append({"r": len(BA), "rA": rA, "wA": weighted_resid(UA, BA), "rB": rB, "rT": rT, "rV": rV, "G": g, "ratio": rT / rB,
                              "p": float(np.mean(gperm >= g)), "gperm_mean": float(gperm.mean())})
            if name == "0.90":
                PB = basis(UB, 0.90)
                angles["A_B"].append(float(np.linalg.svd(BA @ PB.T, compute_uv=False).mean()))
                angles["A_T"].append(float(np.linalg.svd(BA @ PT.T, compute_uv=False).mean()))
    return out, angles


def verdict(vals):
    gpos = np.mean([v["G"] > 0 for v in vals])
    ratio = np.mean([v["ratio"] for v in vals])
    pmed = float(np.median([v["p"] for v in vals]))
    return gpos, ratio, pmed, ("통과" if (gpos >= 0.95 and ratio >= 1.20 and pmed < 0.05) else "불통과")


# ---------------------------------------------------------------- main
def main():
    L = ["# T06b 요약 — coverage gap 보강 진단", ""]
    gate = {"G2": [], "G3_split": [], "G3_rhoA": [], "G3_wA": [], "G3_perm": []}
    b1 = {}  # (ds, k, s) -> {(space, G): acc}
    axis = {}  # (ds, rowname) -> {variant: [60 dicts]}, angles
    d_rows = defaultdict(lambda: defaultdict(list))  # (layer, ds, k) -> method -> list of per-seed arrays
    g1_rows = []

    for ds in DATASETS:
        base_sp = dataset_spaces(ds)
        for s in SEEDS:
            sp_all = dict(base_sp) | rand_spaces(base_sp, s)
            runs = {}
            for layer in LAYERS:
                for k in SHOTS:
                    d = run_dir(layer, ds, k, s)
                    runs[(layer, k)] = {
                        "run": json.load(open(os.path.join(d, "run.json"))),
                        "split": json.load(open(os.path.join(d, "split.json"))),
                        "labels": np.load(os.path.join(d, "labels.npy")),
                        "emb": np.load(os.path.join(d, "emb_best.npy")),
                        "eps": dict(np.load(os.path.join(d, "test_eps.npz"))),
                        "epi": [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl"))],
                    }
            split = runs[(1, 1)]["split"]
            labels = runs[(1, 1)]["labels"]
            for key, r in runs.items():
                gate["G3_split"].append(((ds, s) + key, r["split"] == split and np.array_equal(r["labels"], labels)))
            groups = {"base": split["class_list_train"], "valid": split["class_list_valid"], "test": split["class_list_test"]}
            all_cls = groups["base"] + groups["valid"] + groups["test"]
            by_class = {c: np.where(labels == int(c))[0].tolist() for c in all_cls}

            # ---- G2 + check 3 (same episodes as TEG)
            for (layer, k), r in runs.items():
                e = r["run"]["final"]["best_epoch_valid"]
                m = r["eps"]["epoch"] == e
                sup, qry, cls, epi_idx = r["eps"]["support"][m], r["eps"]["query"][m], r["eps"]["classes"][m], r["eps"]["ep_idx"][m]
                jl = {rec["ep_idx"]: rec for rec in r["epi"] if rec["mode"] == "test" and rec["epoch"] == e}
                ok = (m.sum() == 50 and sup.shape[1] + qry.shape[1] == N_WAY * (k + N_QRY)
                      and all(list(map(float, cls[i])) == list(map(float, jl[int(epi_idx[i])]["classes"])) for i in range(len(epi_idx))))
                gate["G2"].append(((layer, ds, k, s), bool(ok), int(m.sum()), int(sup.shape[1] + qry.shape[1])))
                teg = np.array([jl[int(i)]["acc"] for i in epi_idx])
                d_rows[(layer, ds, k)]["TEG"].append(teg)
                emb = r["emb"]
                methods = {"emb1": emb, "diff1": sp_all["diff1"], "diff2": sp_all["diff2"], "pca1": sp_all["pca1"], "rand1": sp_all["rand1"]} if layer == 1 \
                    else {"emb2": emb, "diff2": sp_all["diff2"], "pca2": sp_all["pca2"], "rand2": sp_all["rand2"]}
                for name, V in methods.items():
                    for metric in ("cos", "euc"):
                        d_rows[(layer, ds, k)][f"{name}·{metric}"].append(probe_acc(V, sup, qry, k, metric))

            # ---- check 1 (probe B1)
            for k in SHOTS:
                rng = random.Random(1000 + s)
                spaces = dict(sp_all) | {"emb1": runs[(1, k)]["emb"], "emb2": runs[(2, k)]["emb"]}
                acc = {}
                for g in GROUPS:
                    eps = episodes(rng, groups[g], by_class, k)
                    for name in SPACES:
                        acc[(name, g)] = fewshot_acc(spaces[name], eps, k)
                b1[(ds, k, s)] = acc
                g1_rows.append((ds, k, s, runs[(1, k)]["run"]["final"]["test_acc_at_best_valid"],
                                json.load(open(os.path.join(R06, f"{ds}_5w{k}s", f"seed{s}", "run.json")))["final"]["test_acc_at_best_valid"]))

            # ---- check 2 (axis loss)
            srng = random.Random(3000 + s)
            base = groups["base"]
            splits = []
            for _ in range(N_SPLIT):
                A_cls = srng.sample(base, len(base) // 2)
                splits.append((A_cls, [c for c in base if c not in A_cls]))
            rows = [(nm, sp_all[nm]) for nm in AXIS_SPACES] + [(f"emb{layer}·{k}shot", runs[(layer, k)]["emb"]) for layer in LAYERS for k in SHOTS]
            for rowname, V in rows:
                protos = {c: l2n(V[by_class[c]].mean(0)) for c in all_cls}
                out, angles = axis_rows(protos, split, splits, s)
                slot = axis.setdefault((ds, rowname), {"v": defaultdict(list), "ang": defaultdict(list)})
                for name in out:
                    slot["v"][name].extend(out[name])
                for a in angles:
                    slot["ang"][a].extend(angles[a])
            print(ds, "seed", s, "done", flush=True)
        del base_sp

    cells = [(ds, k) for ds in DATASETS for k in SHOTS]
    L.append("- run 수: 36 / 36 (층 {1,2} × 3 데이터셋 × 5-way {1,5}-shot × seed 0–2). 값 = seed 3개 평균 ± sd(표 D·D2의 차는 에피소드 150개 평균 ± SE).")
    L.append("- probe: 5-way, K = shot, query 5, cosine 최근접 프로토타입(표 D는 유클리드 제곱 거리도). 정확도 %.")
    L.append("")

    # ---- B1
    L.append("## 표 B1 — 공간 9종 × 클래스 그룹 probe 정확도 (에피소드 500개, %)")
    for g in GROUPS:
        L.append(f"### G = {g}")
        L.append("| 데이터셋 | shot | " + " | ".join(SPACES) + " |")
        L.append("|---|---|" + "---|" * len(SPACES))
        for ds, k in cells:
            L.append(f"| {ds} | {k} | " + " | ".join(mean_sd([b1[(ds, k, s)][(nm, g)] for s in SEEDS])[0] for nm in SPACES) + " |")
        L.append("")

    # ---- B3
    L.append("## 표 B3 — 학습 효과 Δ = acc(emb_h, G) − acc(대조군, G) (%p, seed 평균 ± sd, 부호 일치 수/3)")
    L.append("| 데이터셋 | shot | 비교 | Δ_base | Δ_valid | Δ_test | Δ_base − Δ_test |")
    L.append("|---|---|---|---|---|---|---|")
    p42 = defaultdict(dict)
    for ds, k in cells:
        for h in (1, 2):
            for ctrl in ("rand", "pca"):
                dl = {g: [b1[(ds, k, s)][(f"emb{h}", g)] - b1[(ds, k, s)][(f"{ctrl}{h}", g)] for s in SEEDS] for g in GROUPS}
                gap = [a - b for a, b in zip(dl["base"], dl["test"])]
                txt, m = mean_sd(gap, "{:+.2f}")
                p42[f"emb{h} − {ctrl}{h}"][(ds, k)] = m
                cols = [f"{mean_sd(dl[g], '{:+.2f}')[0]} ({sign_agree(dl[g])}/3)" for g in GROUPS]
                L.append(f"| {ds} | {k} | emb{h} − {ctrl}{h} | " + " | ".join(cols) + f" | {txt} ({sign_agree(gap)}/3) |")
    L.append("")

    # ---- B4
    L.append("## 표 B4 — hop 효과 (2-hop − 1-hop, %p, seed 평균 ± sd)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"{p}·{g}" for p in ("diff2−diff1", "emb2−emb1", "pca2−pca1") for g in GROUPS) + " |")
    L.append("|---|---|" + "---|" * 9)
    p43 = {}
    for ds, k in cells:
        cols = []
        for a, b in (("diff2", "diff1"), ("emb2", "emb1"), ("pca2", "pca1")):
            for g in GROUPS:
                txt, m = mean_sd([b1[(ds, k, s)][(a, g)] - b1[(ds, k, s)][(b, g)] for s in SEEDS], "{:+.2f}")
                cols.append(txt)
                if a == "diff2" and g == "test":
                    p43[(ds, k)] = m
        L.append(f"| {ds} | {k} | " + " | ".join(cols) + " |")
    L.append("")

    # ---- C
    L.append(f"## 표 C — 축 상실 판정 (누적 분산 0.90, base 반분할 {N_SPLIT}회 × seed 3 = 60값, 순열 {N_PERM}회)")
    L.append("판정 = (G > 0 비율 ≥ 0.95) ∧ (평균 ρ̄_T/ρ̄_B ≥ 1.20) ∧ (p 중앙값 < 0.05).")
    L.append("")
    L.append("| 데이터셋 | 공간 | r | ρ̄_A | ρ̄_B | ρ̄_T | ρ̄_V | G | ρ̄_T/ρ̄_B | G > 0 비율 | p 중앙값 | 판정 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    p44 = defaultdict(int)
    rownames = AXIS_SPACES + [f"emb{layer}·{k}shot" for layer in LAYERS for k in SHOTS]
    for ds in DATASETS:
        for rn in rownames:
            vals = axis[(ds, rn)]["v"]["0.90"]
            gpos, ratio, pmed, vd = verdict(vals)
            if rn in ("diff1", "pca1") and vd == "통과":
                p44[rn] += 1
            mv = lambda key, f="{:.4f}": f.format(np.mean([v[key] for v in vals]))
            L.append(f"| {ds} | {rn} | {mv('r', '{:.1f}')} | {mv('rA')} | {mv('rB')} | {mv('rT')} | {mv('rV')} | {mv('G', '{:+.4f}')} | "
                     f"{ratio:.3f} | {gpos:.2f} | {pmed:.3f} | {vd} |")
            rA_mean = float(np.mean([v["rA"] for v in vals]))
            gperm_abs = abs(float(np.mean([v["gperm_mean"] for v in vals])))
            g_sd = float(np.std([v["G"] for v in vals], ddof=1))
            gate["G3_rhoA"].append(((ds, rn), rA_mean, rA_mean <= 0.12))
            wA = [v["wA"] for v in vals]
            gate["G3_wA"].append(((ds, rn), float(np.mean(wA)), float(np.max(wA)), np.mean(wA) <= 0.10 and np.max(wA) <= 0.10))
            gate["G3_perm"].append(((ds, rn), gperm_abs, g_sd, gperm_abs < g_sd))
    L.append("")

    # ---- C-s
    L.append("## 표 C-s — 민감도 (판정만; 괄호: G > 0 비율 / 평균 비율 / p 중앙값)")
    L.append("| 데이터셋 | 공간 | " + " | ".join(name for name, _ in VARIANTS) + " |")
    L.append("|---|---|" + "---|" * len(VARIANTS))
    for ds in DATASETS:
        for rn in rownames:
            cols = []
            for name, _ in VARIANTS:
                gpos, ratio, pmed, vd = verdict(axis[(ds, rn)]["v"][name])
                cols.append(f"{vd} ({gpos:.2f} / {ratio:.2f} / {pmed:.3f})")
            L.append(f"| {ds} | {rn} | " + " | ".join(cols) + " |")
    L.append("")

    # ---- C2
    L.append("## 표 C2 — principal angle 코사인 평균 (0.90 규칙, 분할 20 × seed 3 평균 ± sd)")
    L.append("| 데이터셋 | 공간 | P_A vs P_B | P_A vs P_T |")
    L.append("|---|---|---|---|")
    for ds in DATASETS:
        for rn in rownames:
            ang = axis[(ds, rn)]["ang"]
            L.append(f"| {ds} | {rn} | {mean_sd(ang['A_B'], '{:.4f}')[0]} | {mean_sd(ang['A_T'], '{:.4f}')[0]} |")
    L.append("")

    # ---- D
    L.append("## 표 D — TEG의 실제 test 에피소드(best_valid_epoch, 50개 × seed 3) 위 비교")
    L.append("정확도 = seed 평균 ± sd (%). TEG 대비 차 = 에피소드 150개 짝지은 차 평균 ± SE (%p), 양수 비율 = 차 > 0 비율.")
    L.append("")
    L.append("| 층 | 데이터셋 | shot | 방법·거리 | 정확도 | TEG 대비 차 | 양수 비율 |")
    L.append("|---|---|---|---|---|---|---|")
    p45 = {}
    for layer in LAYERS:
        for ds, k in cells:
            rows = d_rows[(layer, ds, k)]
            teg = np.concatenate(rows["TEG"])
            L.append(f"| L{layer} | {ds} | {k} | TEG | {mean_sd([100 * a.mean() for a in rows['TEG']])[0]} | — | — |")
            for meth, arrs in rows.items():
                if meth == "TEG":
                    continue
                diff = 100 * (np.concatenate(arrs) - teg)
                L.append(f"| L{layer} | {ds} | {k} | {meth} | {mean_sd([100 * a.mean() for a in arrs])[0]} | {mean_se(diff)[0]} | {np.mean(diff > 0):.2f} |")
                if layer == 1 and meth == "diff2·cos":
                    p45[(ds, k)] = (np.mean([a.mean() for a in arrs]), np.mean([a.mean() for a in rows["TEG"]]))
    L.append("")

    # ---- D2
    L.append("## 표 D2 — 분해 (L1, 같은 에피소드 150개): diff1 probe → emb1 probe → TEG (%p, 평균 ± SE, 양수 비율)")
    L.append("| 데이터셋 | shot | 거리 | emb1 − diff1 | TEG − emb1 | TEG − diff1 |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in cells:
        rows = d_rows[(1, ds, k)]
        teg = np.concatenate(rows["TEG"])
        for metric in ("cos", "euc"):
            dfx, emx = np.concatenate(rows[f"diff1·{metric}"]), np.concatenate(rows[f"emb1·{metric}"])
            cols = []
            for a, b in ((emx, dfx), (teg, emx), (teg, dfx)):
                d = 100 * (a - b)
                cols.append(f"{mean_se(d)[0]} ({np.mean(d > 0):.2f})")
            L.append(f"| {ds} | {k} | {metric} | " + " | ".join(cols) + " |")
    L.append("")

    # ---- gates
    g2_fail = [x for x in gate["G2"] if not x[1]]
    g3s_fail = [x[0] for x in gate["G3_split"] if not x[1]]
    g3r_fail = [x for x in gate["G3_rhoA"] if not x[2]]
    g3p_fail = [x for x in gate["G3_perm"] if not x[3]]
    g3w_fail = [x for x in gate["G3_wA"] if not x[3]]
    L.append("## 게이트 점검 기록")
    L.append("| 항목 | 관측 | 결과 |")
    L.append("|---|---|---|")
    L.append(f"| G2 `test_eps.npz` best epoch 에피소드 50개, 노드 수 5·(K+5), classes = episodes.jsonl | {len(gate['G2']) - len(g2_fail)}/{len(gate['G2'])} run 충족 {g2_fail or ''} | {'통과' if not g2_fail else '실패'} |")
    L.append(f"| G3 같은 seed의 split·labels 일치 (L1·L2 × shot 4 run) | {len(gate['G3_split']) - len(g3s_fail)}/{len(gate['G3_split'])} 일치 {g3s_fail or ''} | {'통과' if not g3s_fail else '실패'} |")
    L.append(f"| G3 0.90 규칙 ρ̄_A ≤ 0.12 (행별 60값 평균) — 지시서 원 정의, 사용자 결정으로 아래 줄로 대체 | 최대 {max(x[1] for x in gate['G3_rhoA']):.4f}, 불충족 {[(x[0], round(x[1], 4)) for x in g3r_fail] or '없음'} | 실패(대체됨) |")
    L.append(f"| G3 (사용자 결정, 2026-10-06) 0.90 규칙 분산 가중 잔차 1 − Σ‖P_A d‖²/Σ‖d‖² ≤ 0.10 (A 순서쌍, 행별 60값 평균과 최대) | 행별 평균의 최대 {max(x[1] for x in gate['G3_wA']):.6f}, 행별 최대의 최대 {max(x[2] for x in gate['G3_wA']):.6f}, 충족 {len(gate['G3_wA']) - len(g3w_fail)}/{len(gate['G3_wA'])}행 {[(x[0], round(x[1], 4), round(x[2], 4)) for x in g3w_fail] or ''} | {'통과' if not g3w_fail else '실패'} |")
    L.append(f"| G3 \\|순열 G 평균\\| < 관측 G sd (행별) | 불충족 {[(x[0], round(x[1], 5), round(x[2], 5)) for x in g3p_fail] or '없음'} | {'통과' if not g3p_fail else '실패'} |")
    L.append("")
    L.append("### G1 참고 — L1 `test_acc_at_best_valid` vs T06 같은 seed (일치 요구 없음)")
    L.append("| 데이터셋 | shot | seed | T06b L1 | T06 |")
    L.append("|---|---|---|---|---|")
    for ds, k, s, a, b in g1_rows:
        L.append(f"| {ds} | {k} | {s} | {a:.4f} | {b:.4f} |")
    L.append("")

    # ---- predictions
    n42 = {c: sum(1 for v in p42[c].values() if v > 0) for c in ("emb1 − rand1", "emb1 − pca1")}
    n43 = sum(1 for v in p43.values() if v > 0)
    n45 = sum(1 for a, b in p45.values() if a >= b)
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P42 | 표 B3 (Δ_base − Δ_test) > 0 이 `emb1 − rand1`, `emb1 − pca1` 둘 다에서 ≥ 4/6 | rand1 {n42['emb1 − rand1']}/6, pca1 {n42['emb1 − pca1']}/6 | {'일치' if min(n42.values()) >= 4 else '어긋남'} |")
    L.append(f"| P43 | 표 B4 acc(diff2, test) > acc(diff1, test) ≥ 4/6 | {n43}/6 | {'일치' if n43 >= 4 else '어긋남'} |")
    L.append(f"| P44 | 표 C `diff1`, `pca1` 각각 \"통과\" 데이터셋 ≤ 1/3 | diff1 {p44['diff1']}/3, pca1 {p44['pca1']}/3 | {'일치' if p44['diff1'] <= 1 and p44['pca1'] <= 1 else '어긋남'} |")
    L.append(f"| P45 | 표 D `diff2` cosine probe ≥ TEG(L1) ≥ 3/6 | {n45}/6 | {'일치' if n45 >= 3 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    print("G2 fail", g2_fail, "| G3 split fail", g3s_fail, "| G3 rhoA fail", g3r_fail, "| G3 wA fail", g3w_fail, "| G3 perm fail", g3p_fail)
    print("G3 wA per row (mean, max):", [(x[0], round(x[1], 4), round(x[2], 4)) for x in gate["G3_wA"]])


if __name__ == "__main__":
    main()
