"""T14 R1: --neg_sampler uniform batch indices (first epoch, 50 steps) vs the t13 code path.
Amazon_clothing 1-shot seed 0, --pres_loss infonce --pres_lambda 10 --pres_pool nb --pres_m 1024. No training: the trainer is built
(seed_everything(0) first, as main.py) and only the sampling function is called 50 times. t13 model.py is read with `git show t13:model.py`."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from argument import parse_args  # noqa: E402
from utils import seed_everything  # noqa: E402


def build(module, seed=0):
    sys.argv = ["x", "--dataset", "Amazon_clothing", "--way", "5", "--shot", "1", "--seed", str(seed), "--device", "0",
                "--pres_lambda", "10", "--pres_pool", "nb", "--pres_loss", "infonce", "--pres_m", "1024", "--neg_sampler", "uniform"]
    args, _ = parse_args()
    conf = yaml.safe_load(open("configuration.yaml"))
    seed_everything(seed)
    return module.teg_trainer(args, conf, seed)


def main():
    tmp = os.path.join(os.environ.get("T14_SCRATCH", "/tmp"), "model_t13.py")
    open(tmp, "w").write(subprocess.check_output(["git", "show", "t13:model.py"], cwd=ROOT).decode())
    spec = importlib.util.spec_from_file_location("model_t13", tmp)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    import model as new  # noqa: E402
    t_old = build(old)
    a = [t_old._pres_sample().cpu().numpy() for _ in range(50)]
    t_new = build(new)
    b = [t_new._pres_batch()[0].cpu().numpy() for _ in range(50)]
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    print(f"R1 steps 50, batch size {len(a[0])}, pool {len(t_new.pres_pool_idx)}; identical: {same}; "
          f"first 5 of step 0: {a[0][:5].tolist()} / {b[0][:5].tolist()}")


if __name__ == "__main__":
    main()
