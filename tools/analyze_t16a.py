"""T16a gates G1-G6, G8 and the D / J tables -> results/T16a/gates.json, results/T16a/tables.md (copied into the report)."""
import json
import math
import os
import random
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import erasure_metrics as em  # noqa: E402

RES = os.path.join(ROOT, "results", "T16a")
EP_KEYS = ("epoch", "ep_idx", "mode", "support", "query", "classes")


def P(name):
    return os.path.join(RES, name)


def glog(name):
    return json.load(open(os.path.join(P(name), "gate_log.json")))


def run(name):
    return json.load(open(os.path.join(P(name), "run.json")))


def fixed_test_acc(name):
    Z = np.load(os.path.join(P(name), "fixed_eval_logits.npz"))
    m = Z["mode"] == 1
    return 100 * float((Z["logits"][m].argmax(-1) == Z["query_y"][m]).mean())


def npz_eq(a, b, f):
    A, B = np.load(os.path.join(P(a), f)), np.load(os.path.join(P(b), f))
    return all(np.array_equal(A[k], B[k]) for k in EP_KEYS)


def first_update_losses(L):
    v = [x["value"] for x in L["nll"] if x["mode"] == "train" and x["epoch"] == 1]
    return v[0], v[1]


def accs(name):
    r = run(name)["final"]
    return {"best_acc_valid": 100 * r["best_acc_valid"], "test_acc_at_best_valid": 100 * r["test_acc_at_best_valid"], "fixed_test": fixed_test_acc(name)}


def compare(ref, new, o1, o2, eval_only=False):
    """G1-style comparison of `new` against `ref`; tolerance from old1/old2."""
    Lr, Ln, L1, L2 = glog(ref), glog(new), glog(o1), glog(o2)
    out = {}
    if eval_only:
        out["rng_equal"] = Lr["rng"] == Ln["rng"]
        ev = lambda L: [e for e in L["episodes"] if e["mode"] != "train"]
        out["episodes_equal"] = ev(Lr) == ev(Ln)
    else:
        out["rng_equal"] = Lr["rng"] == Ln["rng"]
        out["episodes_equal"] = Lr["episodes"] == Ln["episodes"]
    out["eval_logits_episodes_equal"] = npz_eq(ref, new, "eval_logits.npz")
    out["fixed_episodes_equal"] = npz_eq(ref, new, "fixed_eval_logits.npz")
    a = (out["rng_equal"] and out["episodes_equal"] and out["eval_logits_episodes_equal"] and out["fixed_episodes_equal"])
    lr_, ln_, l1_, l2_ = first_update_losses(Lr), first_update_losses(Ln), first_update_losses(L1), first_update_losses(L2)
    tol_l = [max(abs(l2_[i] - l1_[i]), 1e-6) for i in range(2)]
    out["first_update_L_N_L_G"] = {"ref": lr_, "new": ln_, "tol": tol_l}
    b = all(abs(ln_[i] - lr_[i]) <= tol_l[i] for i in range(2))
    ar, an, a1, a2 = accs(ref), accs(new), accs(o1), accs(o2)
    tol_a = {k: max(abs(a2[k] - a1[k]), 0.1) for k in ar}
    out["accs"] = {"ref": ar, "new": an, "tol": tol_a}
    c = all(abs(an[k] - ar[k]) <= tol_a[k] for k in ar)
    out["abc"] = [a, b, c]
    out["pass"] = a and b and c
    return out


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def main():
    G = {}
    # ---------------- G1, G2
    for ds, k in (("Amazon_clothing", 1), ("dblp", 5)):
        o1, o2 = f"old/old1_{ds}_{k}", f"old/old2_{ds}_{k}"
        G[f"G1_{ds}_{k}"] = compare(o1, f"G1_new_{ds}_{k}", o1, o2)
        G[f"G2_{ds}_{k}"] = compare(f"G1_new_{ds}_{k}", f"G2_hooks_{ds}_{k}", o1, o2)
        G[f"old1_vs_old2_{ds}_{k}"] = {"rng_equal": glog(o1)["rng"] == glog(o2)["rng"], "episodes_equal": glog(o1)["episodes"] == glog(o2)["episodes"]}
    # ---------------- G3
    import torch  # noqa: F401

    from analyze_t13 import groups_of
    from analyze_t14 import knn_purity as kp14
    from analyze_t06b import episodes as ep06, fewshot_acc as fs06

    d = os.path.join(ROOT, "results", "T14", "uni", "Amazon_clothing_5w1s", "seed0")
    Z = np.load(os.path.join(d, "emb_best.npy"))
    grp, sp, lab = groups_of(d)
    nodes, y = grp["test"]
    a14, aem = kp14(Z, nodes, y), em.knn_purity(Z, nodes, y)
    by_class = {c: np.where(lab == int(c))[0].tolist() for c in sp["class_list_test"]}
    e_em = em.episodes(random.Random(1000), sp["class_list_test"], by_class, 1, 500)
    e_06 = ep06(random.Random(1000), sp["class_list_test"], by_class, 1)
    same_eps = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(e_em, e_06)) and len(e_em) == len(e_06)
    G["G3a"] = {"t14_knn_purity": a14, "em_knn_purity": aem, "absdiff": abs(a14 - aem), "pass": abs(a14 - aem) <= 1e-6}
    G["G3b"] = {"episodes_equal_500": same_eps, "t06b_fewshot": fs06(Z, e_06, 1), "em_fewshot": em.fewshot_acc(Z, e_em, 1),
                "em_fewshot_200": em.fewshot_acc(Z, e_em[:200], 1), "t06b_fewshot_200": fs06(Z, e_06[:200], 1)}
    G["G3b"]["pass"] = same_eps and G["G3b"]["t06b_fewshot"] == G["G3b"]["em_fewshot"] and G["G3b"]["em_fewshot_200"] == G["G3b"]["t06b_fewshot_200"]
    # (c) offline recompute of the G2 run step-500 M1/M3 from traj_z_ends.npz
    sys.path.insert(0, ROOT)
    cwd = os.getcwd()
    os.chdir(ROOT)
    from utils import load_data, seed_everything

    g3c = {}
    for ds, k in (("Amazon_clothing", 1), ("dblp", 5)):
        seed_everything(0)
        _, _, _, labels, _, cl_tr, cl_va, cl_te, _, _ = load_data(ds)
        labt = labels.numpy()
        T = json.load(open(os.path.join(P(f"G2_hooks_{ds}_{k}"), "traj.json")))
        last = T["records"][-1]
        z = np.load(os.path.join(P(f"G2_hooks_{ds}_{k}"), "traj_z_ends.npz"))[f"z_step{last['step']}"]
        groups, byc = {}, {}
        for g, cl in (("base", cl_tr), ("valid", cl_va), ("test", cl_te)):
            nd = np.where(np.isin(labt, [int(c) for c in cl]))[0]
            groups[g] = (list(cl), nd, labt[nd])
            for c in cl:
                byc[c] = np.where(labt == int(c))[0].tolist()
        rng = random.Random(1000)
        eps = {g: em.episodes(rng, groups[g][0], byc, k, 200) for g in ("base", "valid", "test")}
        diffs = {}
        for g in ("base", "valid", "test"):
            diffs[f"purity_{g}"] = abs(em.knn_purity(z, groups[g][1], groups[g][2]) - last[f"purity_{g}"])
            diffs[f"probe_{g}"] = abs(em.fewshot_acc(z, eps[g], k) - last[f"probe_{g}"])
        g3c[f"{ds}_{k}"] = {"step": last["step"], "absdiff": diffs, "max": max(diffs.values())}
    os.chdir(cwd)
    G["G3c"] = g3c | {"pass": all(v["max"] <= 1e-6 for v in g3c.values())}
    # ---------------- G4
    rc = json.load(open(P("relabel_check.json")))
    g4 = {}
    for ds, ref in (("Amazon_clothing", "G1_new_Amazon_clothing_1"), ("dblp", "G4_default_dblp_1")):
        new = f"G4_relabel_{ds}_1"
        Lr, Ln = glog(ref), glog(new)
        ev = lambda L: [e for e in L["episodes"] if e["mode"] != "train"]
        tr = lambda L: [e for e in L["episodes"] if e["mode"] == "train"]
        a = {"rng_equal": Lr["rng"] == Ln["rng"], "valid_test_episodes_equal": ev(Lr) == ev(Ln),
             "eval_logits_episodes_equal": npz_eq(ref, new, "eval_logits.npz"), "fixed_episodes_equal": npz_eq(ref, new, "fixed_eval_logits.npz"),
             "train_classes_equal": [e["classes"] for e in tr(Lr)] == [e["classes"] for e in tr(Ln)],
             "train_nodes_equal": [e["support"] + e["query"] for e in tr(Lr)] == [e["support"] + e["query"] for e in tr(Ln)]}
        b = {x: rc[ds][x] for x in ("class_lists_equal", "base_sizes_equal", "base_node_set_equal", "valid_test_lists_equal", "labels_equal")}
        g4[ds] = {"a": a, "b": b, "c_majority_frac_mean": rc[ds]["fake_class_majority_frac_mean"],
                  "c_majority_frac_min_max": [min(rc[ds]["fake_class_majority_frac"]), max(rc[ds]["fake_class_majority_frac"])],
                  "base_lists_changed": rc[ds]["base_lists_changed"], "n_base_classes": rc[ds]["n_base_classes"],
                  "pass": all(a[x] for x in ("rng_equal", "valid_test_episodes_equal", "eval_logits_episodes_equal", "fixed_episodes_equal")) and all(b.values())}
    G["G4"] = g4 | {"pass": all(v["pass"] for v in g4.values())}
    # ---------------- G5
    gp = lambda n: json.load(open(os.path.join(P(n), "grad_probe.json")))
    g5 = {}
    for ds in ("Amazon_clothing", "dblp", "Amazon_electronics"):
        r1 = gp(f"D2_g1_{ds}_1")
        g5[f"a_{ds}"] = max(v for r in r1 for v in r["rel_sup_vs_gammaLN"].values())
        r0 = gp(f"D2_g0_{ds}_1")
        g5[f"b_{ds}"] = {"gcn_gamma_L_N_max": max(v for r in r0 for v in r["gcn"]["gamma_L_N"].values()),
                         "egnn_sup_max": max(r["egnn"]["sup"] for r in r0)}
    r3 = gp("D3_sup0_Amazon_clothing_1")
    g5["c"] = {"gcn_sup_max": max(v for r in r3 for v in r["gcn"]["sup"].values()), "egnn_sup_max": max(r["egnn"]["sup"] for r in r3)}
    G["G5"] = g5 | {"pass": all(g5[f"a_{d}"] <= 1e-6 for d in ("Amazon_clothing", "dblp", "Amazon_electronics"))
                    and all(g5[f"b_{d}"]["gcn_gamma_L_N_max"] == 0 and g5[f"b_{d}"]["egnn_sup_max"] == 0 for d in ("Amazon_clothing", "dblp", "Amazon_electronics"))
                    and g5["c"]["gcn_sup_max"] == 0 and g5["c"]["egnn_sup_max"] == 0}
    # ---------------- G6 finite
    bad = []
    runs = [json.loads(l) for l in open(P("_runs.jsonl"))]
    for r in runs:
        n = r["name"]
        if r["exit"] != 0:
            bad.append((n, "exit"))
            continue
        L = glog(n)
        if not all(math.isfinite(x["value"]) for x in L["nll"]):
            bad.append((n, "nll"))
        if not finite(run(n)["final"]):
            bad.append((n, "final"))
        for f in ("traj.json", "grad_probe.json"):
            if os.path.exists(os.path.join(P(n), f)) and not finite(json.load(open(os.path.join(P(n), f)))):
                bad.append((n, f))
    G["G6"] = {"n_runs": len(runs), "bad": bad, "pass": not bad}
    # ---------------- J4, J5 (dropout tap), J6
    J4 = glog("J4_dumpemb_Amazon_clothing_1")["dump_emb_epochs"]
    G["J4"] = {"dump_embedding_epochs": J4, "epoch0_called": 0 in J4, "emb_epoch": run("J4_dumpemb_Amazon_clothing_1").get("emb_epoch")}
    fd = glog("G1_new_Amazon_clothing_1")["first_train_dropout"]
    G["J5_first_train_dropout"] = fd | {"cpu_changed": fd["rng_before"]["cpu"] != fd["rng_after"]["cpu"],
                                        "cuda_changed": fd["rng_before"]["cuda"] != fd["rng_after"]["cuda"]}
    a, b = glog("G1_new_Amazon_clothing_1"), glog("J6_gcn128_Amazon_clothing_1")
    G["J6"] = {"rng_equal_per_epoch": a["rng"] == b["rng"], "episodes_equal": a["episodes"] == b["episodes"],
               "fixed_episodes_equal": npz_eq("G1_new_Amazon_clothing_1", "J6_gcn128_Amazon_clothing_1", "fixed_eval_logits.npz"),
               "first_dropout_mask_equal": a["first_train_dropout"]["mask_sha1"] == b["first_train_dropout"]["mask_sha1"],
               "first_dropout_rng_before_equal": a["first_train_dropout"]["rng_before"] == b["first_train_dropout"]["rng_before"]}
    # ---------------- G8 timing
    sec = {r["name"]: r["sec"] for r in runs}
    G["G8"] = {k: sec[k] for k in sec if k.startswith(("G8_", "G1_new", "G2_hooks", "old/"))}
    json.dump(G, open(P("gates.json"), "w"), indent=1, default=str)
    for k, v in G.items():
        print(k, json.dumps(v, default=str)[:400])


if __name__ == "__main__":
    main()
