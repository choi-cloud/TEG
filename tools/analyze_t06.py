"""T06 analysis: results/T06/**/{run.json, emb_best.npy, split.json, labels.npy} -> reports/T06_summary.md.

Reads saved files only (plus the dataset via utils.load_data for the raw/diff spaces). Labels are used for
analysis only. Analysis randomness: one random.Random(1000 + seed) instance per run.
"""
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from utils import load_data  # noqa: E402

R06 = os.path.join(ROOT, "results", "T06")
R04 = os.path.join(ROOT, "results", "T04")
OUT = os.path.join(ROOT, "reports", "T06_summary.md")

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
SPACES = ["raw", "diff", "emb"]
GROUPS = ["base", "valid", "test"]
N_WAY, N_QRY, N_EPI, N_RAND = 5, 5, 500, 1000
BATCH = 50


def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def l2n(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def dataset_spaces(ds):
    # raw: X as returned by load_data; diff: A_hat^2 X with A_hat = GCNConv's normalization (self-loops, sym norm)
    edges, _, X, _, _, _, _, _, _, n = load_data(ds)
    X = X.numpy().astype(np.float32)
    ei, w = gcn_norm(edges, None, X.shape[0], add_self_loops=True)
    ei, w = ei.numpy(), w.numpy()
    A = sp.csr_matrix((w, (ei[1], ei[0])), shape=(X.shape[0], X.shape[0]))  # out[target] += w * x[source]
    D = (A @ (A @ X)).astype(np.float32)
    return {"raw": X, "diff": D}


def episodes(rng, classes, by_class, k):
    out = []
    for _ in range(N_EPI):
        cls = rng.sample(classes, N_WAY)
        sup, qry = [], []
        for c in cls:
            ids = rng.sample(by_class[c], k + N_QRY)
            sup.append(ids[:k])
            qry.append(ids[k:])
        out.append((np.array(sup), np.array(qry)))
    return out


def fewshot_acc(V, eps):
    accs = []
    for b in range(0, len(eps), BATCH):
        sup = np.stack([e[0] for e in eps[b : b + BATCH]])  # (B, N, K)
        qry = np.stack([e[1] for e in eps[b : b + BATCH]])  # (B, N, Q)
        P = l2n(V[sup].mean(2))  # (B, N, d)
        Q = l2n(V[qry].reshape(sup.shape[0], -1, V.shape[1]))  # (B, N*Q, d)
        pred = np.einsum("bqd,bnd->bqn", Q, P).argmax(2)
        true = np.repeat(np.arange(N_WAY), N_QRY)[None, :]
        accs.extend((pred == true).mean(1).tolist())
    return float(np.mean(accs))


def coverage(emb, labels, split, rng):
    protos = {c: l2n(emb[labels == int(c)].mean(0)) for c in split["class_list_train"] + split["class_list_valid"] + split["class_list_test"]}
    base = split["class_list_train"]
    Dm = np.stack([protos[a] - protos[b] for a in base for b in base if a != b])  # ordered pairs: mean is zero
    _, s, Vt = np.linalg.svd(Dm, full_matrices=False)
    ratio = np.cumsum(s**2) / np.sum(s**2)
    r = int(np.searchsorted(ratio, 0.9) + 1)
    B = Vt[:r]

    def resid(dirs):
        proj = dirs @ B.T @ B
        return float(np.mean(np.sum((dirs - proj) ** 2, 1) / np.sum(dirs**2, 1)))

    def pairs(cl):
        return np.stack([protos[cl[i]] - protos[cl[j]] for i in range(len(cl)) for j in range(i + 1, len(cl))])

    rand = np.array([[rng.gauss(0.0, 1.0) for _ in range(emb.shape[1])] for _ in range(N_RAND)])
    return {"r": r, "test": resid(pairs(split["class_list_test"])), "valid": resid(pairs(split["class_list_valid"])), "rand": resid(rand)}


def main():
    rows = {}
    for ds in DATASETS:
        spaces = dataset_spaces(ds)
        for k in SHOTS:
            for s in SEEDS:
                d = os.path.join(R06, f"{ds}_5w{k}s", f"seed{s}")
                run = json.load(open(os.path.join(d, "run.json")))
                split = json.load(open(os.path.join(d, "split.json")))
                labels = np.load(os.path.join(d, "labels.npy"))
                emb = np.load(os.path.join(d, "emb_best.npy"))
                ref = json.load(open(os.path.join(R04, f"{ds}_5w{k}s", f"seed{s}", "run.json")))
                rng = random.Random(1000 + s)
                by_class = {c: np.where(labels == int(c))[0].tolist() for c in split["class_list_train"] + split["class_list_valid"] + split["class_list_test"]}
                groups = {"base": split["class_list_train"], "valid": split["class_list_valid"], "test": split["class_list_test"]}
                acc = {}
                for g in GROUPS:
                    eps = episodes(rng, groups[g], by_class, k)  # same episodes for all 3 spaces
                    for sp_name, V in (("raw", spaces["raw"]), ("diff", spaces["diff"]), ("emb", emb)):
                        acc[(sp_name, g)] = 100 * fewshot_acc(V, eps)
                e = run["final"]["best_epoch_valid"]
                rec = run["epochs"][e]
                rows[(ds, k, s)] = {
                    "acc": acc,
                    "A": (100 * rec["train_acc"], 100 * rec["valid_acc"], 100 * rec["test_acc"]),
                    "C": coverage(emb, labels, split, rng),
                    "best_epoch": e,
                    "emb_epoch": run["emb_epoch"],
                    "t06": run["final"]["test_acc_at_best_valid"],
                    "t04": ref["final"]["test_acc_at_best_valid"],
                    "n_cls": tuple(len(groups[g]) for g in GROUPS),
                }
                print(ds, k, s, "done", flush=True)

    cells = [(ds, k) for ds in DATASETS for k in SHOTS]
    L = ["# T06 요약 — coverage gap 진단", ""]
    L.append(f"- run 수: **{len(rows)} / 18** (3 데이터셋 × 5-way {{1,5}}-shot × seed 0–2). 값 = (데이터셋, shot)별 seed 3개 평균 ± sd.")
    L.append("- 분석용 난수: run마다 `random.Random(1000 + seed)` 한 개(에피소드 base → valid → test 순, 이어서 무작위 방향).")
    L.append("")

    L.append("## 표 A — 일반화 격차 (off `best_valid_epoch`의 정확도, %)")
    L.append("| 데이터셋 | shot | best_valid_epoch (seed순) | train | valid | test | train − test | valid − test |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in cells:
        v = [rows[(ds, k, s)] for s in SEEDS]
        tr, va, te = ([x["A"][i] for x in v] for i in range(3))
        L.append(f"| {ds} | {k} | {', '.join(str(x['best_epoch']) for x in v)} | {mean_sd(tr)[0]} | {mean_sd(va)[0]} | {mean_sd(te)[0]} | "
                 f"{mean_sd([a - c for a, c in zip(tr, te)], '{:+.2f}')[0]} | {mean_sd([b - c for b, c in zip(va, te)], '{:+.2f}')[0]} |")
    L.append("")

    L.append(f"## 표 B1 — 공간별 few-shot 분리도 (cosine 최근접 프로토타입, 5-way, K = shot, query 5, 에피소드 {N_EPI}개, 정확도 %)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"{sp_}·{g}" for sp_ in SPACES for g in GROUPS) + " |")
    L.append("|---|---|" + "---|" * (len(SPACES) * len(GROUPS)))
    p41 = {}
    for ds, k in cells:
        v = [rows[(ds, k, s)] for s in SEEDS]
        cols = {(sp_, g): mean_sd([x["acc"][(sp_, g)] for x in v]) for sp_ in SPACES for g in GROUPS}
        p41[(ds, k)] = (cols[("emb", "test")][1], cols[("diff", "test")][1])
        L.append(f"| {ds} | {k} | " + " | ".join(cols[(sp_, g)][0] for sp_ in SPACES for g in GROUPS) + " |")
    L.append("")

    L.append("## 표 B2 — Δ_G = acc(`emb`, G) − acc(`diff`, G) (%p)")
    L.append("| 데이터셋 | shot | Δ_base | Δ_valid | Δ_test | Δ_base − Δ_test |")
    L.append("|---|---|---|---|---|---|")
    p40 = {}
    for ds, k in cells:
        v = [rows[(ds, k, s)] for s in SEEDS]
        dl = {g: [x["acc"][("emb", g)] - x["acc"][("diff", g)] for x in v] for g in GROUPS}
        gap = mean_sd([b - t for b, t in zip(dl["base"], dl["test"])], "{:+.2f}")
        p40[(ds, k)] = gap[1]
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd(dl[g], "{:+.2f}")[0] for g in GROUPS) + f" | {gap[0]} |")
    L.append("")

    L.append("## 표 C — 구분 축 coverage (`emb`, base 프로토타입 차이 PCA 90% 부분공간 P, 잔차 비율 ‖(I−P)d‖²/‖d‖²)")
    L.append(f"| 데이터셋 | shot | r | 잔차(test 쌍) | 잔차(valid 쌍) | 잔차(무작위 {N_RAND}) | (base/valid/test 클래스 수) |")
    L.append("|---|---|---|---|---|---|---|")
    for ds, k in cells:
        v = [rows[(ds, k, s)] for s in SEEDS]
        L.append(f"| {ds} | {k} | {mean_sd([x['C']['r'] for x in v], '{:.1f}')[0]} | {mean_sd([x['C']['test'] for x in v], '{:.4f}')[0]} | "
                 f"{mean_sd([x['C']['valid'] for x in v], '{:.4f}')[0]} | {mean_sd([x['C']['rand'] for x in v], '{:.4f}')[0]} | "
                 f"{'/'.join(map(str, v[0]['n_cls']))} |")
    L.append("")

    L.append("## 참고 — G1·G2 기록")
    L.append("| 데이터셋 | shot | seed | `emb_epoch` | `best_epoch_valid` | 일치 | T06 `test_acc_at_best_valid` | T04 같은 seed |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, k in cells:
        for s in SEEDS:
            x = rows[(ds, k, s)]
            L.append(f"| {ds} | {k} | {s} | {x['emb_epoch']} | {x['best_epoch']} | {'예' if x['emb_epoch'] == x['best_epoch'] else '아니오'} | {x['t06']:.4f} | {x['t04']:.4f} |")
    L.append("")

    n40 = sum(1 for v in p40.values() if v > 0)
    n41 = sum(1 for e, d in p41.values() if e < d)
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P40 | 표 B2 (Δ_base − Δ_test) > 0 인 (데이터셋, shot) ≥ 4/6 | {n40}/6 (" + ", ".join(f"{p40[c]:+.2f}" for c in cells) + f") | {'일치' if n40 >= 4 else '어긋남'} |")
    L.append(f"| P41 | 표 B1 acc(`emb`, test) < acc(`diff`, test) 인 (데이터셋, shot) ≥ 2/6 | {n41}/6 | {'일치' if n41 >= 2 else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
