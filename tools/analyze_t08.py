"""T08 analysis (E0, dual-view combination) -> reports/T08_summary.md.

Inputs: results/T08/<ds>_5w<k>s/seed<s>/{run.json, episodes.jsonl, eval_logits.npz}, T06b L1 run.json (G1 record),
and the dataset via utils.load_data for view D (A_hat^2 X, row-normalized; A_hat as in T06b).
Labels are used for analysis only. Analysis RNG: query half split random.Random(5000 + seed) (one instance per run).
"""
import json
import math
import os
import random
import sys

import numpy as np
import scipy.sparse as sp
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from utils import load_data  # noqa: E402
from analyze_t06b import probe_acc  # noqa: E402  (T06b cosine probe, used for gate G3)

R08 = os.path.join(ROOT, "results", "T08")
R6B = os.path.join(ROOT, "results", "T06b", "L1")
OUT = os.path.join(ROOT, "reports", "T08_summary.md")

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SHOTS = [1, 5]
SEEDS = [0, 1, 2]
N_WAY, N_QRY = 5, 5
TL_GRID = 10.0 ** np.round(np.arange(-2.0, 4.0 + 1e-9, 0.2), 1)  # 31 values
S_GRID = 10.0 ** np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 1)  # 31 values
A21 = np.round(np.arange(0, 21) * 0.05, 2)
A5 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


# ---------------------------------------------------------------- helpers
def mean_sd(xs, fmt="{:.2f}"):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return f"{fmt.format(m)} ± {fmt.replace('+', '').format(sd)}", m


def paired(diff_by_seed):
    """diff_by_seed: list (seeds) of per-episode diff arrays in %p. Returns text, mean, se."""
    d = np.concatenate(diff_by_seed)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    sg = (m > 0) - (m < 0)
    agree = sum(1 for x in diff_by_seed if sg != 0 and ((float(x.mean()) > 0) - (float(x.mean()) < 0)) == sg)
    return f"{m:+.2f} ± {se:.2f} ({np.mean(d > 0):.2f}, {agree}/{len(diff_by_seed)})", m, se


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def nll(logp, y):
    return float(-np.take_along_axis(logp, y[..., None], -1).mean())


def acc_per_ep(logp, y):
    return (logp.argmax(-1) == y).mean(-1)


def diffusion_view(ds):
    edges, _, X, _, _, _, _, _, _, _ = load_data(ds)
    X = X.numpy().astype(np.float32)
    ei, w = gcn_norm(edges, None, X.shape[0], add_self_loops=True)
    ei, w = ei.numpy(), w.numpy()
    A = sp.csr_matrix((w, (ei[1], ei[0])), shape=(X.shape[0], X.shape[0]))  # same A_hat as T06b
    D2 = (A @ (A @ X)).astype(np.float32)
    return D2 / np.maximum(np.linalg.norm(D2, axis=1, keepdims=True), 1e-12)


def view_d_scores(H, sup, qry, k):
    """cos(h_q, mean of support h) -> (E, N*Q, N)."""
    E = sup.shape[0]
    P = H[sup].reshape(E, N_WAY, k, H.shape[1]).mean(2)
    P = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
    Q = H[qry]
    return np.einsum("eqd,end->eqn", Q, P).astype(np.float64)


def combine(lpL, lpD, a):
    return log_softmax(a * lpL + (1 - a) * lpD)


# ---------------------------------------------------------------- main
def main():
    runs = {}
    gate = {"G2": [], "G3": []}
    for ds in DATASETS:
        H = diffusion_view(ds)
        for k in SHOTS:
            for s in SEEDS:
                d = os.path.join(R08, f"{ds}_5w{k}s", f"seed{s}")
                run = json.load(open(os.path.join(d, "run.json")))
                epi = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl"))]
                Z = dict(np.load(os.path.join(d, "eval_logits.npz")))
                jl = {(r["epoch"], r["mode"], r["ep_idx"]): r for r in epi}

                # ---- G2: dump consistency over all dumped episodes
                acc_dump = acc_per_ep(Z["logits"], Z["query_y"])
                mism = 0
                for i in range(len(Z["epoch"])):
                    mode = "valid" if Z["mode"][i] == 0 else "test"
                    rec = jl[(int(Z["epoch"][i]), mode, int(Z["ep_idx"][i]))]
                    if float(acc_dump[i]) != rec["acc"] or list(map(float, Z["classes"][i])) != list(map(float, rec["classes"])):
                        mism += 1
                vmean_mism = 0
                for e in run["epochs"]:
                    for mi, key in ((0, "valid_acc"), (1, "test_acc")):
                        m = (Z["epoch"] == e["epoch"]) & (Z["mode"] == mi)
                        if abs(float(acc_dump[m].mean()) - e[key]) > 1e-12:
                            vmean_mism += 1
                gate["G2"].append(((ds, k, s), len(Z["epoch"]), mism, vmean_mism))

                # ---- target episodes: best_epoch_valid, valid 50 / test 50 (ordered by ep_idx)
                e = run["final"]["best_epoch_valid"]
                sel = {}
                for mi, mode in ((0, "valid"), (1, "test")):
                    m = np.where((Z["epoch"] == e) & (Z["mode"] == mi))[0]
                    m = m[np.argsort(Z["ep_idx"][m])]
                    sel[mode] = {
                        "L": Z["logits"][m].astype(np.float64),
                        "D": view_d_scores(H, Z["support"][m], Z["query"][m], k),
                        "y": Z["query_y"][m].astype(int),
                        "sup": Z["support"][m],
                        "qry": Z["query"][m],
                        "teg": np.array([jl[(e, mode, int(i))]["acc"] for i in Z["ep_idx"][m]]),
                    }

                # ---- temperatures on valid (mean NLL over all valid queries)
                V = sel["valid"]
                nll_T = [nll(log_softmax(V["L"] / t), V["y"]) for t in TL_GRID]
                nll_s = [nll(log_softmax(sv * V["D"]), V["y"]) for sv in S_GRID]
                iT, iS = int(np.argmin(nll_T)), int(np.argmin(nll_s))
                TL, sD = float(TL_GRID[iT]), float(S_GRID[iS])
                lp = {mode: (log_softmax(sel[mode]["L"] / TL), log_softmax(sD * sel[mode]["D"])) for mode in sel}

                # ---- alpha_val: argmin valid NLL over A21, ties -> closer to 1
                nll_a = [nll(combine(*lp["valid"], a), V["y"]) for a in A21]
                best = min(nll_a)
                a_val = float(max(a for a, v in zip(A21, nll_a) if v == best))

                T = sel["test"]
                curve = {float(a): acc_per_ep(combine(*lp["test"], a), T["y"]) for a in A21}
                lpL_t, lpD_t = lp["test"]

                # ---- G3
                g3_epcount = (len(sel["valid"]["y"]), len(T["y"]))
                g3_a1 = bool(np.array_equal(curve[1.0], T["teg"]))
                probe = probe_acc(H, T["sup"], T["qry"], k, "cos")
                g3_a0 = bool(np.array_equal(curve[0.0], probe))
                gate["G3"].append(((ds, k, s), g3_epcount, g3_a1, g3_a0))

                # ---- table C: query half split (12 / 13), one split per episode
                rng = random.Random(5000 + s)
                adapt, fixed = [], []
                for i in range(len(T["y"])):
                    h1 = sorted(rng.sample(range(N_WAY * N_QRY), 12))
                    h2 = [q for q in range(N_WAY * N_QRY) if q not in h1]
                    accs = {}
                    for a in list(A5) + [a_val]:
                        pred = combine(lpL_t[i], lpD_t[i], a).argmax(-1)
                        accs[float(a)] = ((pred[h1] == T["y"][i][h1]).mean(), (pred[h2] == T["y"][i][h2]).mean())

                    def pick(half):
                        top = max(accs[float(a)][half] for a in A5)
                        cands = [float(a) for a in A5 if accs[float(a)][half] == top]
                        return min(cands, key=lambda a: (abs(a - a_val), -a))

                    adapt.append((accs[pick(0)][1] + accs[pick(1)][0]) / 2)
                    fixed.append((accs[a_val][0] + accs[a_val][1]) / 2)

                # ---- table D: single-view predictions (before combination)
                cL = lpL_t.argmax(-1) == T["y"]
                cD = lpD_t.argmax(-1) == T["y"]
                runs[(ds, k, s)] = {
                    "e": e, "TL": TL, "s": sD, "iT": iT, "iS": iS, "a_val": a_val, "curve": curve,
                    "adapt": np.array(adapt), "fixed": np.array(fixed),
                    "D": {"L만": float((cL & ~cD).mean()), "D만": float((~cL & cD).mean()), "둘 다 정답": float((cL & cD).mean()),
                          "둘 다 오답": float((~cL & ~cD).mean()), "하나라도 정답": float((cL | cD).mean())},
                    "t08": run["final"]["test_acc_at_best_valid"],
                    "t06b": json.load(open(os.path.join(R6B, f"{ds}_5w{k}s", f"seed{s}", "run.json")))["final"]["test_acc_at_best_valid"],
                    "probe_rawdiff2_equal": None,
                }
                print(ds, k, s, "done", flush=True)
        del H

    cells = [(ds, k) for ds in DATASETS for k in SHOTS]
    L = ["# T08 요약 — E0: 두 관점 결합 여지 (TEG logit × 확산 코사인)", ""]
    L.append(f"- run 수: {len(runs)} / 18 (3 데이터셋 × 5-way {{1,5}}-shot × seed 0–2, 1층 GCN). 대상: run마다 `best_epoch_valid`의 valid 50·test 50 에피소드.")
    L.append("- 정확도 %, seed 3개 평균 ± sd. 짝지은 차: 에피소드 150개 평균 ± SE (양수 비율, seed 부호 일치 수/3), %p.")
    L.append("- 관점 L 점수 = 덤프 텐서(`model.py`의 `output = output_softmax.cpu().detach()`, `accuracy()` 입력). 관점 D 점수 = cos(h̄_q, support h̄ 평균), h̄ = 행 정규화 Â²X.")
    L.append("")

    # ---- Table A
    L.append("## 표 A — 결합 곡선 (test, α ∈ A21). max·argmax는 test로 고른 값(상한 참고용)")
    L.append("| 데이터셋 | shot | " + " | ".join(f"{a:.2f}" for a in A21) + " | acc(α=0) | acc(α=1) | max | argmax α | max − max(끝점) |")
    L.append("|---|---|" + "---|" * (len(A21) + 5))
    p46 = {}
    curve_mean = {}
    for ds, k in cells:
        per_a = {float(a): [100 * runs[(ds, k, s)]["curve"][float(a)].mean() for s in SEEDS] for a in A21}
        means = {a: sum(v) / len(v) for a, v in per_a.items()}
        curve_mean[(ds, k)] = means
        mx = max(means.values())
        am = max(a for a, v in means.items() if v == mx)
        gain = mx - max(means[0.0], means[1.0])
        p46[(ds, k)] = gain
        L.append(f"| {ds} | {k} | " + " | ".join(f"{means[float(a)]:.2f}" for a in A21) +
                 f" | {mean_sd(per_a[0.0])[0]} | {mean_sd(per_a[1.0])[0]} | {mx:.2f} | {am:.2f} | {gain:+.2f} |")
    L.append("")

    # ---- Table B
    L.append("## 표 B — valid로 고른 α (α_val = argmin valid NLL, 동률이면 1에 가까운 값)")
    L.append("| 데이터셋 | shot | α_val (seed순) | α_val 평균 | T_L (seed순) | s (seed순) | test acc(α_val) | 차 vs TEG(α=1) | 차 vs D(α=0) | 나은 끝점 | 차 vs 나은 끝점 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    j1, p47, p49 = 0, {}, {}
    for ds, k in cells:
        rr = [runs[(ds, k, s)] for s in SEEDS]
        acc_val = [r["curve"][r["a_val"]] for r in rr]
        d1 = [100 * (a - r["curve"][1.0]) for a, r in zip(acc_val, rr)]
        d0 = [100 * (a - r["curve"][0.0]) for a, r in zip(acc_val, rr)]
        better = 1.0 if curve_mean[(ds, k)][1.0] >= curve_mean[(ds, k)][0.0] else 0.0
        db = [100 * (a - r["curve"][better]) for a, r in zip(acc_val, rr)]
        tb, mb, seb = paired(db)
        if mb > 0 and mb > 2 * seb:
            j1 += 1
        p47[(ds, k)] = mb
        am = sum(r["a_val"] for r in rr) / len(rr)
        p49[(ds, k)] = am
        av_txt = ", ".join("{:.2f}".format(r["a_val"]) for r in rr)
        tl_txt = ", ".join("{:g}".format(r["TL"]) for r in rr)
        s_txt = ", ".join("{:g}".format(r["s"]) for r in rr)
        L.append(f"| {ds} | {k} | {av_txt} | {am:.3f} | {tl_txt} | "
                 f"{s_txt} | {mean_sd([100 * a.mean() for a in acc_val])[0]} | {paired(d1)[0]} | {paired(d0)[0]} | "
                 f"{'TEG(α=1)' if better == 1.0 else 'D(α=0)'} | {tb} |")
    L.append("")

    # ---- Table C
    L.append("## 표 C — task별 선택의 여지 (test query 12/13 반분할 1회, α ∈ A5, 양방향 평균)")
    L.append("| 데이터셋 | shot | acc(적응) | acc(고정 α_val) | 적응 − 고정 | J2 조건 충족 |")
    L.append("|---|---|---|---|---|---|")
    j2 = 0
    for ds, k in cells:
        rr = [runs[(ds, k, s)] for s in SEEDS]
        dd = [100 * (r["adapt"] - r["fixed"]) for r in rr]
        txt, m, se = paired(dd)
        ok = m >= 1.0 and m > 2 * se
        j2 += ok
        L.append(f"| {ds} | {k} | {mean_sd([100 * r['adapt'].mean() for r in rr])[0]} | {mean_sd([100 * r['fixed'].mean() for r in rr])[0]} | {txt} | {'예' if ok else '아니오'} |")
    L.append("")

    # ---- Table D
    keys = ["L만", "D만", "둘 다 정답", "둘 다 오답", "하나라도 정답"]
    L.append("## 표 D — query 단위 상보성 (test, 결합 전 단독 예측, 비율 %)")
    L.append("| 데이터셋 | shot | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for ds, k in cells:
        rr = [runs[(ds, k, s)] for s in SEEDS]
        L.append(f"| {ds} | {k} | " + " | ".join(mean_sd([100 * r["D"][key] for r in rr])[0] for key in keys) + " |")
    L.append("")

    # ---- J
    L.append("## 판정 (§3.3, 값만)")
    L.append("| 판정 | 정의 | 값 |")
    L.append("|---|---|---|")
    L.append(f"| J1 결합 가치 | 표 B 차 vs 나은 끝점: 평균 > 0 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {j1}/6 |")
    L.append(f"| J2 task별 판단 가치 | 표 C 적응 − 고정: 평균 ≥ 1.0%p 이고 평균 > 2·SE 인 (데이터셋, shot) 수 | {j2}/6 |")
    L.append("")

    # ---- gates & records
    g2_fail = [g for g in gate["G2"] if g[2] or g[3]]
    g3_fail = [g for g in gate["G3"] if not (g[1] == (50, 50) and g[2] and g[3])]
    L.append("## 게이트 점검 기록")
    L.append("| 항목 | 관측 | 결과 |")
    L.append("|---|---|---|")
    L.append(f"| G2 덤프 에피소드별 정확도·classes = episodes.jsonl (valid·test, 전 에폭), 에폭별 평균 = run.json | "
             f"덤프 에피소드 {sum(g[1] for g in gate['G2'])}개, 불일치 {sum(g[2] for g in gate['G2'])}, 에폭 평균 불일치 {sum(g[3] for g in gate['G2'])} | {'통과' if not g2_fail else '실패'} |")
    L.append(f"| G3 대상 valid 50·test 50, α=1 에피소드별 = TEG, α=0 에피소드별 = T06b `probe_acc`(cos, h̄) | "
             f"{18 - len(g3_fail)}/18 run 충족 {[g[0] for g in g3_fail] or ''} | {'통과' if not g3_fail else '실패'} |")
    L.append("")
    edge = [((ds, k, s), "T_L", r["TL"]) for (ds, k, s), r in runs.items() if r["iT"] in (0, len(TL_GRID) - 1)] + \
           [((ds, k, s), "s", r["s"]) for (ds, k, s), r in runs.items() if r["iS"] in (0, len(S_GRID) - 1)]
    L.append(f"- 격자 끝 최적값: {edge or '없음'}")
    L.append("")
    L.append("### G1 참고 — `test_acc_at_best_valid` vs T06b L1 같은 seed (일치 요구 없음)")
    L.append("| 데이터셋 | shot | seed | T08 | T06b L1 | best_valid_epoch |")
    L.append("|---|---|---|---|---|---|")
    for ds, k in cells:
        for s in SEEDS:
            r = runs[(ds, k, s)]
            L.append(f"| {ds} | {k} | {s} | {r['t08']:.4f} | {r['t06b']:.4f} | {r['e']} |")
    L.append("")

    # ---- predictions
    n46 = sum(1 for v in p46.values() if v >= 1.0)
    n47 = sum(1 for v in p47.values() if v > 0)
    p49_ok = all(p49[("dblp", k)] >= 0.5 for k in SHOTS) and all(p49[("Amazon_clothing", k)] <= 0.5 for k in SHOTS)
    L.append("## 사전 등록 대비")
    L.append("| 예측 | 내용 | 관측 | 일치·어긋남 |")
    L.append("|---|---|---|---|")
    L.append(f"| P46 | 표 A max − max(끝점) ≥ 1.0%p 인 (데이터셋, shot) ≥ 4/6 | {n46}/6 (" + ", ".join(f"{p46[c]:+.2f}" for c in cells) + f") | {'일치' if n46 >= 4 else '어긋남'} |")
    L.append(f"| P47 | 표 B 차 vs 나은 끝점 평균 > 0 인 (데이터셋, shot) ≥ 4/6 | {n47}/6 (" + ", ".join(f"{p47[c]:+.2f}" for c in cells) + f") | {'일치' if n47 >= 4 else '어긋남'} |")
    L.append(f"| P48 | J2 ≤ 2/6 | J2 = {j2}/6 | {'일치' if j2 <= 2 else '어긋남'} |")
    L.append(f"| P49 | α_val seed 평균: dblp 두 shot ≥ 0.5, Amazon_clothing 두 shot ≤ 0.5 | dblp {p49[('dblp', 1)]:.3f} / {p49[('dblp', 5)]:.3f}, "
             f"Amazon_clothing {p49[('Amazon_clothing', 1)]:.3f} / {p49[('Amazon_clothing', 5)]:.3f} | {'일치' if p49_ok else '어긋남'} |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    print("G2 fail", g2_fail, "| G3 fail", g3_fail, "| grid edge", edge)


if __name__ == "__main__":
    main()
