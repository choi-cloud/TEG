"""T16a G4(b)(c): build the trainer (no training) with --relabel_base 0 and 1 (seed 0) and compare id_by_class / labels.
Writes results/T16a/relabel_check.json."""
import json
import os
import sys
import warnings
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings(action="ignore")
torch.set_num_threads(3)
import model  # noqa: E402
from argument import parse_args  # noqa: E402
from utils import seed_everything  # noqa: E402


def build(ds, relabel):
    sys.argv = ["main.py", "--dataset", ds, "--way", "5", "--shot", "1", "--seed", "0", "--device", "0", "--relabel_base", str(relabel)]
    args, _ = parse_args()
    seed_everything(0)
    return model.teg_trainer(args, yaml.safe_load(open("configuration.yaml")), 0)


out = {}
for ds in ("Amazon_clothing", "dblp"):
    a, b = build(ds, 0), build(ds, 1)
    base = list(a.class_list_train)
    other = list(a.class_list_valid) + list(a.class_list_test)
    lab = a.labels.cpu().numpy()
    purity = []
    for c in base:
        cnt = Counter(int(x) for x in lab[b.id_by_class[c]])
        purity.append(max(cnt.values()) / len(b.id_by_class[c]))
    out[ds] = {
        "class_lists_equal": base == list(b.class_list_train) and list(a.class_list_valid) == list(b.class_list_valid)
        and list(a.class_list_test) == list(b.class_list_test),
        "base_sizes_equal": [len(a.id_by_class[c]) for c in base] == [len(b.id_by_class[c]) for c in base],
        "base_node_set_equal": set(n for c in base for n in a.id_by_class[c]) == set(n for c in base for n in b.id_by_class[c]),
        "base_lists_changed": sum(a.id_by_class[c] != b.id_by_class[c] for c in base),
        "n_base_classes": len(base),
        "valid_test_lists_equal": all(a.id_by_class[c] == b.id_by_class[c] for c in other),
        "labels_equal": bool(np.array_equal(lab, b.labels.cpu().numpy())),
        "structural_equal": bool(torch.equal(a.structural_features, b.structural_features)),
        "fake_class_majority_frac": purity, "fake_class_majority_frac_mean": float(np.mean(purity)),
        "relabel_label_matches_lists": all(int(b.relabel_label[n]) == int(c) for c in base for n in b.id_by_class[c]),
    }
    print(ds, {k: v for k, v in out[ds].items() if k != "fake_class_majority_frac"}, flush=True)
    del a, b
json.dump(out, open("results/T16a/relabel_check.json", "w"), indent=1)
