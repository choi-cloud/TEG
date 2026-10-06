"""T08b gate checks G2/G3. Runs tools/analyze_t08b.py (writes reports/T08b_summary.md) and compares against T08:

- T08 per-run reference values are recomputed with the functions of tools/analyze_t08.py (the T08 code path).
- prob-mode tables rendered by analyze_t08b are compared line by line with the committed reports/T08_summary.md.
Output: printed and written to results/T08b/gate_checks.txt.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_t08 as T8  # noqa: E402  (T08 code path, reference only)
import analyze_t08b as T8B  # noqa: E402

ROOT = T8B.ROOT
OUT = os.path.join(ROOT, "results", "T08b", "gate_checks.txt")


def t08_reference(H, d, k):
    """Replicates the per-run part of analyze_t08.main with its own functions."""
    run = json.load(open(os.path.join(d, "run.json")))
    Z = dict(np.load(os.path.join(d, "eval_logits.npz")))
    epi = {(r["epoch"], r["mode"], r["ep_idx"]): r for r in map(json.loads, open(os.path.join(d, "episodes.jsonl")))}
    e = run["final"]["best_epoch_valid"]
    sel = {}
    for mi, mode in ((0, "valid"), (1, "test")):
        m = np.where((Z["epoch"] == e) & (Z["mode"] == mi))[0]
        m = m[np.argsort(Z["ep_idx"][m])]
        sel[mode] = {"L": Z["logits"][m].astype(np.float64), "D": T8.view_d_scores(H, Z["support"][m], Z["query"][m], k),
                     "y": Z["query_y"][m].astype(int), "teg": np.array([epi[(e, mode, int(i))]["acc"] for i in Z["ep_idx"][m]])}
    V = sel["valid"]
    iT = int(np.argmin([T8.nll(T8.log_softmax(V["L"] / t), V["y"]) for t in T8.TL_GRID]))
    iS = int(np.argmin([T8.nll(T8.log_softmax(sv * V["D"]), V["y"]) for sv in T8.S_GRID]))
    TL, sD = float(T8.TL_GRID[iT]), float(T8.S_GRID[iS])
    lp = {mode: (T8.log_softmax(sel[mode]["L"] / TL), T8.log_softmax(sD * sel[mode]["D"])) for mode in sel}
    nll_a = [T8.nll(T8.combine(*lp["valid"], a), V["y"]) for a in T8.A21]
    a_val = float(max(a for a, v in zip(T8.A21, nll_a) if v == min(nll_a)))
    T = sel["test"]
    acc = lambda a: T8.acc_per_ep(T8.combine(*lp["test"], a), T["y"])
    return {"TL": TL, "s": sD, "a_val": a_val, "acc_val": acc(a_val), "acc1": acc(1.0), "acc0": acc(0.0), "teg": T["teg"]}


def table_rows(path, header):
    lines = open(path).read().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith(header))
    rows = []
    for l in lines[i + 1:]:
        if l.startswith("## ") or (rows and not l.startswith("|")):
            break
        if l.startswith("|"):
            rows.append(l)
    return rows


def main():
    runs_log, runs_prob, sec, sec_p = T8B.main()
    out = []
    g2_ep, g3_run = [], []
    for ds in T8B.DATASETS:
        H = T8.diffusion_view(ds)
        for k in T8B.SHOTS:
            for s in T8B.SEEDS:
                ref = t08_reference(H, T8B.run_dir(ds, k, s), k)
                rl, rp = runs_log[(ds, k, s)], runs_prob[(ds, k, s)]
                g2_ep.append(((ds, k, s), np.array_equal(rl["curve"][1.0], ref["acc1"]), np.array_equal(rl["curve"][0.0], ref["acc0"]),
                              np.array_equal(rl["curve"][1.0], ref["teg"]), rl["argmax_same"], rl["n_replaced"]))
                g3_run.append(((ds, k, s), rp["a_val"] == ref["a_val"], rp["TL"] == ref["TL"], rp["s"] == ref["s"],
                               np.array_equal(rp["curve"][rp["a_val"]], ref["acc_val"])))
        del H
    t08 = os.path.join(ROOT, "reports", "T08_summary.md")
    heads = {"A": "## 표 A", "B": "## 표 B", "C": "## 표 C", "D": "## 표 D", "J": "## 판정"}
    eq = {key: table_rows(t08, h) == sec_p[key] for key, h in heads.items()}
    eqD_log = table_rows(t08, "## 표 D") == sec["D"]

    out.append("G2 per run: (run, alpha=1 == T08 alpha=1, alpha=0 == T08 alpha=0, alpha=1 == TEG episodes.jsonl, argmax(log p)==argmax(p), log(0) replaced)")
    out += [str(x) for x in g2_ep]
    out.append(f"G2 summary: alpha1 {sum(x[1] for x in g2_ep)}/18, alpha0 {sum(x[2] for x in g2_ep)}/18, TEG {sum(x[3] for x in g2_ep)}/18, "
               f"argmax {sum(x[4] for x in g2_ep)}/18, replaced total {sum(x[5] for x in g2_ep)}, table D (log) == T08 table D: {eqD_log}")
    out.append("G3 per run: (run, alpha_val, T_L, s, per-episode test acc(alpha_val)) equal to T08 code path with l_transform='prob'")
    out += [str(x) for x in g3_run]
    out.append(f"G3 summary: {sum(all(x[1:]) for x in g3_run)}/18 runs all equal; prob-mode rendered rows == T08_summary.md rows: {eq}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(out) + "\n")
    print("\n".join(out[-1:] + [out[len(g2_ep) + 1]]))


if __name__ == "__main__":
    main()
