"""T21 S1: MLP on row-normalized A_hat^2 X trained with cross-entropy on all base-class nodes (full batch, no episodes).

Input  fusion_baseline.diffusion_features_hops(ds, 2). Base classes / labels: T09 `base` seed s split.json / labels.npy.
Model  Linear(d, 64) -> ReLU -> Linear(64, |base|). Parameters are drawn directly (no nn.Linear, whose constructor
       consumes the torch default generator) from torch.Generator().manual_seed(16000 + seed) on the CPU,
       order W1, b1, W2, b2, each uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)).
Train  Adam lr 5e-3, weight decay 5e-4, 1000 epochs. Snapshot h = ReLU(W1 x + b1) of all nodes after e steps,
       e in {0, 25, 50, 100, 200, 500, 1000} -> results/T21/mlp/<ds>/seed<s>/h_ep<e>.npy; base training accuracy and loss after e steps,
       e = 0-1000 -> run.json.
Usage  CUDA_VISIBLE_DEVICES=<UUID> python tools/ladder_t21.py
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import fusion_baseline as fb  # noqa: E402

DATASETS = ["Amazon_clothing", "dblp", "Amazon_electronics"]
SEEDS = [0, 1, 2]
SNAP = [0, 25, 50, 100, 200, 500, 1000]
EPOCHS, LR, WD, HID = 1000, 5e-3, 5e-4, 64  # 2026-10-14: 200 -> 1000 (T21 Hc(1) addendum)
R09 = os.path.join(ROOT, "results", "T09", "base")
OUT = os.path.join(ROOT, "results", "T21", "mlp")


def init_params(d, c, seed):
    g = torch.Generator().manual_seed(16000 + seed)
    ps = []
    for shape, fan_in in (((HID, d), d), ((HID,), d), ((c, HID), HID), ((c,), HID)):
        b = 1.0 / math.sqrt(fan_in)
        ps.append(torch.empty(shape).uniform_(-b, b, generator=g))
    return ps


def main():
    dev = torch.device("cuda:0")
    for ds in DATASETS:
        X = torch.from_numpy(fb.diffusion_features_hops(ds, 2)).to(dev)
        for s in SEEDS:
            t0 = time.time()
            d9 = os.path.join(R09, f"{ds}_5w1s", f"seed{s}")
            sp = json.load(open(os.path.join(d9, "split.json")))
            labels = np.load(os.path.join(d9, "labels.npy"))
            base = [int(c) for c in sp["class_list_train"]]
            cmap = {c: i for i, c in enumerate(base)}
            nodes = np.where(np.isin(labels, base))[0]
            idx = torch.from_numpy(nodes).to(dev)
            y = torch.tensor([cmap[int(c)] for c in labels[nodes]], device=dev)
            W1, b1, W2, b2 = [p.to(dev).requires_grad_() for p in init_params(X.shape[1], len(base), s)]
            opt = torch.optim.Adam([W1, b1, W2, b2], lr=LR, weight_decay=WD)
            od = os.path.join(OUT, ds, f"seed{s}")
            os.makedirs(od, exist_ok=True)
            acc, loss_log = [], []
            for e in range(EPOCHS + 1):
                h = F.relu(F.linear(X, W1, b1))
                logits = F.linear(h[idx], W2, b2)
                acc.append(100 * float((logits.argmax(1) == y).float().mean()))
                if e in SNAP:
                    np.save(os.path.join(od, f"h_ep{e}.npy"), h.detach().cpu().numpy().astype(np.float32))
                loss = F.cross_entropy(logits, y)
                loss_log.append(float(loss))
                if e == EPOCHS:
                    break
                opt.zero_grad()
                loss.backward()
                opt.step()
            json.dump({"dataset": ds, "seed": s, "n_base_classes": len(base), "n_base_nodes": int(len(nodes)), "d": int(X.shape[1]),
                       "hidden": HID, "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "init_generator": 16000 + s,
                       "init": "uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)), order W1 b1 W2 b2",
                       "base_train_acc": acc, "loss": loss_log, "sec": time.time() - t0},
                      open(os.path.join(od, "run.json"), "w"), indent=1)
            print(ds, s, f"acc ep0 {acc[0]:.2f} ep200 {acc[-1]:.2f} sec {time.time() - t0:.1f}", flush=True)
        del X
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
