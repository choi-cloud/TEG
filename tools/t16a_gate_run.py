"""T16a gate wrapper: one in-process training run of the TEG code found at <code_root> (old worktree or the repo), with
read-only taps (no RNG use):
  - hash of random.getstate() at every train_epoch call (mode, epoch)
  - every sampled episode (task_generator_in_class: support, query, classes) in call order
  - every NLLLoss value with (mode, epoch, call index)  -> first update step L_N, L_G
  - dump_embedding call epochs (J4)
Writes <out_dir>/gate_log.json and the usual run outputs.
Usage: python tools/t16a_gate_run.py <code_root> <out_dir> <seed> -- <main.py args without --seed/--out_dir>
The working directory is the repo root (dataset/ and configuration.yaml), the modules come from <code_root>.
"""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_root, out_dir, seed = os.path.abspath(sys.argv[1]), sys.argv[2], int(sys.argv[3])
extra = sys.argv[sys.argv.index("--") + 1 :]
sys.path.insert(0, code_root)
os.chdir(REPO)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings(action="ignore")
torch.set_num_threads(3)

import model  # noqa: E402
from argument import parse_args  # noqa: E402
from utils import seed_everything  # noqa: E402

assert os.path.dirname(os.path.abspath(model.__file__)) == code_root, model.__file__
ctx = {"mode": None, "epoch": None}
log = {"code_root": code_root, "rng": [], "episodes": [], "nll": [], "dump_emb_epochs": []}

orig_tg = model.task_generator_in_class


def tg(*a, **k):
    s, q, c = orig_tg(*a, **k)
    log["episodes"].append({"mode": ctx["mode"], "epoch": ctx["epoch"], "support": [int(x) for x in s], "query": [int(x) for x in q],
                            "classes": [int(x) for x in c]})
    return s, q, c


model.task_generator_in_class = tg
orig_te = model.teg_trainer.train_epoch


def te(self, mode, n_episode, epoch):
    ctx["mode"], ctx["epoch"] = mode, epoch
    log["rng"].append({"mode": mode, "epoch": epoch, "sha1": hashlib.sha1(repr(random.getstate()).encode()).hexdigest()})
    return orig_te(self, mode, n_episode, epoch)


model.teg_trainer.train_epoch = te
orig_de = model.teg_trainer.dump_embedding


def de(self, epoch):
    log["dump_emb_epochs"].append(int(epoch))
    return orig_de(self, epoch)


model.teg_trainer.dump_embedding = de
OrigNLL = torch.nn.NLLLoss


class NLL(OrigNLL):
    def forward(self, inp, target):
        out = super().forward(inp, target)
        log["nll"].append({"mode": ctx["mode"], "epoch": ctx["epoch"], "value": float(out.detach())})
        return out


torch.nn.NLLLoss = NLL
orig_do = torch.nn.functional.dropout
log["first_train_dropout"] = None


def h_state():
    return {"cpu": hashlib.sha1(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda": hashlib.sha1(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()}


def do(inp, p=0.5, training=True, inplace=False):
    if training and log["first_train_dropout"] is None:
        before = h_state()
        out = orig_do(inp, p, training, inplace)
        after = h_state()
        mask = (out != 0).cpu().numpy()
        log["first_train_dropout"] = {"epoch": ctx["epoch"], "mode": ctx["mode"], "device": str(inp.device),
                                      "mask_sha1": hashlib.sha1(np.packbits(mask).tobytes()).hexdigest(),
                                      "rng_before": before, "rng_after": after}
        return out
    return orig_do(inp, p, training, inplace)


torch.nn.functional.dropout = do
sys.argv = ["main.py", "--seed", str(seed), "--num_seed", "1", "--device", "0", "--out_dir", out_dir] + extra
args, _ = parse_args()
seed_everything(seed)
t = model.teg_trainer(args, yaml.safe_load(open(os.path.join(REPO, "configuration.yaml"))), seed)
t.train()
os.makedirs(out_dir, exist_ok=True)
json.dump(log, open(os.path.join(out_dir, "gate_log.json"), "w"))
print("done", out_dir)
