"""T07 gate checks G1-G4 -> printed and results/T07/gate_checks.txt.

G1: episodes (support/query/classes, all epochs, valid+test) identical across the 7 configs of a (dataset, shot, seed);
    base test_acc_at_best_valid next to T08 (record only).
G2: model teacher (model.diffusion_teacher) vs fusion_baseline.diffusion_features; fixed episodes identical across configs;
    checkpoint re-evaluation (run.json ckpt_recheck) and ckpt_epoch == best_epoch_valid; pres_loss_mean finite for lambda > 0.
G3: files. G4: extended fusion_baseline reproduces t08b table B (per run vs the t08b version of the file, and rendered rows).
"""
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import fusion_baseline as fb  # noqa: E402
import analyze_t07 as A7  # noqa: E402

OUT = os.path.join(ROOT, "results", "T07", "gate_checks.txt")
EP_KEYS = ("epoch", "ep_idx", "mode", "support", "query", "classes")


def main():
    out = []
    # ---- G3 files
    files = ("run.json", "eval_logits.npz", "fixed_eval_logits.npz", "emb_best.npy")
    miss = [(c, ds, k, s) for c in A7.CONFIGS for ds in A7.DATASETS for k in A7.SHOTS for s in A7.SEEDS
            if not all(os.path.exists(os.path.join(A7.run_dir(c, ds, k, s), f)) for f in files)]
    out.append(f"G3 files: {126 - len(miss)}/126 complete; missing {miss}")

    # ---- G1 / G2 per (ds, k, s)
    g1_bad, g2_fixed_bad, g2_ck_bad, g2_pres_bad, base_vs_t08 = [], [], [], [], []
    for ds in A7.DATASETS:
        for k in A7.SHOTS:
            for s in A7.SEEDS:
                ref = ref_f = None
                for c in A7.CONFIGS:
                    d = A7.run_dir(c, ds, k, s)
                    if (c, ds, k, s) in miss:
                        continue
                    Z = np.load(os.path.join(d, "eval_logits.npz"))
                    F = np.load(os.path.join(d, "fixed_eval_logits.npz"))
                    run = json.load(open(os.path.join(d, "run.json")))
                    if ref is None:
                        ref, ref_f = {x: Z[x] for x in EP_KEYS}, {x: F[x] for x in EP_KEYS}
                    else:
                        if not all(np.array_equal(ref[x], Z[x]) for x in EP_KEYS):
                            g1_bad.append((c, ds, k, s))
                        if not all(np.array_equal(ref_f[x], F[x]) for x in EP_KEYS):
                            g2_fixed_bad.append((c, ds, k, s))
                    ck = run["ckpt_recheck"]
                    if not (ck and ck["n"] == 50 and ck["n_equal"] == 50 and run["ckpt_epoch"] == run["final"]["best_epoch_valid"]):
                        g2_ck_bad.append(((c, ds, k, s), ck, run["ckpt_epoch"], run["final"]["best_epoch_valid"]))
                    if A7.LAM[c] > 0:
                        vals = [e.get("pres_loss_mean") for e in run["epochs"]]
                        if not all(v is not None and math.isfinite(v) for v in vals):
                            g2_pres_bad.append((c, ds, k, s))
                    if c == "base":
                        t08 = json.load(open(os.path.join(ROOT, "results", "T08", f"{ds}_5w{k}s", f"seed{s}", "run.json")))
                        base_vs_t08.append(((ds, k, s), run["final"]["test_acc_at_best_valid"], t08["final"]["test_acc_at_best_valid"]))
    out.append(f"G1 episodes identical across 7 configs (all epochs, valid+test): mismatched runs {g1_bad or 'none'}")
    out.append(f"G1 base vs T08 test_acc_at_best_valid: equal {sum(a == b for _, a, b in base_vs_t08)}/{len(base_vs_t08)}; " +
               "; ".join(f"{key}: {a:.4f} / {b:.4f}" for key, a, b in base_vs_t08))
    out.append(f"G2 fixed episodes identical across configs: mismatched {g2_fixed_bad or 'none'}")
    out.append(f"G2 checkpoint re-evaluation 50/50 and ckpt_epoch == best_epoch_valid: failures {g2_ck_bad or 'none'}")
    out.append(f"G2 pres_loss_mean finite at every epoch (lambda > 0): failures {g2_pres_bad or 'none'}")

    # ---- G2 teacher vs fusion_baseline
    import torch  # noqa: F401
    from utils import load_data
    import model as M

    for ds in A7.DATASETS:
        edges, _, X, _, _, _, _, _, _, _ = load_data(ds)
        t_model = M.diffusion_teacher(X, edges)
        t_fb = fb.diffusion_features(ds)
        out.append(f"G2 teacher {ds}: max |model - fusion_baseline| = {float(np.abs(t_model - t_fb).max()):.3e}")

    # ---- G4 fusion_baseline regression vs t08b
    src = subprocess.run(["git", "show", "t08b:tools/fusion_baseline.py"], capture_output=True, text=True, check=True).stdout
    tmp = os.path.join(ROOT, "results", "T07", "_fusion_baseline_t08b.py")
    open(tmp, "w").write(src)
    spec = importlib.util.spec_from_file_location("fb_t08b", tmp)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    import analyze_t08b as T8B

    ok = 0
    for ds in T8B.DATASETS:
        H = fb.diffusion_features(ds)
        for k in T8B.SHOTS:
            for s in T8B.SEEDS:
                a = fb.evaluate(T8B.run_dir(ds, k, s), H, "log", "val")
                b = old.evaluate(T8B.run_dir(ds, k, s), H, "log", "val")
                ok += a["alpha"] == b["alpha"] and a["T_L"] == b["T_L"] and a["s"] == b["s"] and np.array_equal(a["acc"], b["acc"])
    runs_log = T8B.compute_runs("log")
    secB = T8B.render(runs_log)[0]["B"]
    lines = open(os.path.join(ROOT, "reports", "T08b_summary.md")).read().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith("## 표 B —"))
    t08b_rows = [l for l in lines[i + 1:i + 9] if l.startswith("|")]
    out.append(f"G4 per-run (alpha_val, T_L, s, per-episode test acc) current vs t08b fusion_baseline: {ok}/18 equal; "
               f"rendered table B rows == reports/T08b_summary.md: {secB == t08b_rows}")
    os.remove(tmp)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
