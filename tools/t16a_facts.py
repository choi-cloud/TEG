"""T16a J1/J2/J3/J5 (no training). Seed 0, default flags. Writes results/T16a/facts.json.
J1 F per dataset; J2 nodes absent from the .mat files (label left 0), how many fall in the nb pool, size of id_by_class[0];
J3 all-zero rows of structural_features; J5 torch CPU / CUDA RNG state hashes around a train-mode GCN forward, outside and
inside torch.random.fork_rng(devices=[device])."""
import hashlib
import json
import os
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings(action="ignore")
torch.set_num_threads(3)
import model  # noqa: E402
from argument import parse_args  # noqa: E402
from utils import seed_everything  # noqa: E402


def h():
    return {"cpu": hashlib.sha1(torch.get_rng_state().numpy().tobytes()).hexdigest()[:12],
            "cuda": hashlib.sha1(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()[:12]}


out = {}
for ds in ("Amazon_clothing", "dblp", "Amazon_electronics"):
    sys.argv = ["main.py", "--dataset", ds, "--way", "5", "--shot", "1", "--seed", "0", "--device", "0"]
    args, _ = parse_args()
    seed_everything(0)
    t = model.teg_trainer(args, yaml.safe_load(open("configuration.yaml")), 0)
    r = {"F": int(t.features.shape[1]), "N": int(t.features.shape[0]), "conv1.lin.weight": list(t.conv.conv1.lin.weight.shape)}
    tr = sio.loadmat(f"./dataset/{ds}/{ds}_train.mat")["Index"].ravel()
    te = sio.loadmat(f"./dataset/{ds}/{ds}_test.mat")["Index"].ravel()
    in_mat = np.zeros(t.features.shape[0], dtype=bool)
    in_mat[tr] = True
    in_mat[te] = True
    absent = np.where(~in_mat)[0]
    lab = t.labels.cpu().numpy()
    base = set(int(c) for c in t.class_list_train)
    keys0 = [k for k in t.id_by_class if float(k) == 0.0]
    r["J2"] = {"n_absent_from_mat": int(len(absent)), "absent_label_values": sorted(set(int(x) for x in lab[absent])),
               "absent_in_nb_pool": int(sum(int(lab[i]) not in base for i in absent)),
               "id_by_class_0_size": int(len(t.id_by_class[keys0[0]])) if keys0 else None,
               "class0_in_train_list": 0 in base, "class0_in_valid_list": 0 in set(int(c) for c in t.class_list_valid),
               "class0_in_test_list": 0 in set(int(c) for c in t.class_list_test),
               "n_train_index": int(len(tr)), "n_test_index": int(len(te))}
    sf = t.structural_features.cpu().numpy()
    r["J3"] = {"zero_rows": int((np.abs(sf).sum(1) == 0).sum()), "shape": list(sf.shape)}
    # J5
    t.conv.train()
    s0 = h()
    _ = t.conv(t.features, t.edges)
    s1 = h()
    with torch.random.fork_rng(devices=[t.device]):
        _ = t.conv(t.features, t.edges)
        s2_in = h()
    s2 = h()
    r["J5"] = {"before": s0, "after_train_forward": s1, "inside_fork_after_forward": s2_in, "after_fork": s2,
               "forward_changes_cpu": s0["cpu"] != s1["cpu"], "forward_changes_cuda": s0["cuda"] != s1["cuda"],
               "fork_restores_cpu": s1["cpu"] == s2["cpu"], "fork_restores_cuda": s1["cuda"] == s2["cuda"]}
    out[ds] = r
    print(ds, json.dumps(r), flush=True)
    del t
    torch.cuda.empty_cache()
os.makedirs("results/T16a", exist_ok=True)
json.dump(out, open("results/T16a/facts.json", "w"), indent=1)
