"""T14 R2: global-RNG check. One short in-process run (--epochs 2) with the episode generator wrapped to log every sampled
training / valid / test episode (classes, support, query) in call order; saved to <out_dir>/episodes_all.npz.
Usage: python tools/check_t14_r2.py <out_dir> <neg_sampler>   (Amazon_clothing 1-shot seed 0, infonce lambda 10)."""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from argument import parse_args  # noqa: E402
from utils import seed_everything  # noqa: E402
import model  # noqa: E402

out_dir, sampler = sys.argv[1], sys.argv[2]
log = []
orig = model.task_generator_in_class


def wrapped(*a, **k):
    s, q, c = orig(*a, **k)
    log.append((np.array(s), np.array(q), [int(x) for x in c]))
    return s, q, c


model.task_generator_in_class = wrapped
sys.argv = ["x", "--dataset", "Amazon_clothing", "--way", "5", "--shot", "1", "--seed", "0", "--device", "0", "--epochs", "2",
            "--pres_lambda", "10", "--pres_pool", "nb", "--pres_loss", "infonce", "--neg_sampler", sampler,
            "--dump_logits", "--fixed_eval", "--out_dir", out_dir]
args, _ = parse_args()
seed_everything(0)
t = model.teg_trainer(args, yaml.safe_load(open("configuration.yaml")), 0)
t.train()
np.savez_compressed(os.path.join(out_dir, "episodes_all.npz"), support=np.stack([x[0] for x in log]), query=np.stack([x[1] for x in log]),
                    classes=np.array([x[2] for x in log]))
print(json.dumps({"n_episodes_logged": len(log), "bsc": [e.get("bsc") for e in json.load(open(os.path.join(out_dir, "run.json")))["epochs"]]}))
