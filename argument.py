import argparse
from ast import parse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    # _______________
    # General Setting
    timestr = time.strftime("%m%d-%H%M")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)  # 50
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, default="ogbn-arxiv", choices=["Amazon_clothing", "Amazon_electronics", "corafull", "dblp", "coauthorCS", "ogbn-arxiv"]
    )
    parser.add_argument("--data_dir", type=str, default="datasets", help="dir of datasets")
    parser.add_argument("--num_seed", type=int, default=1)
    parser.add_argument("--summary", type=str, default=timestr)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--way", type=int, default=5)  # N
    parser.add_argument("--shot", type=int, default=3)  # K
    parser.add_argument("--qry", type=int, default=5)  # M
    parser.add_argument("--episodes", type=int, default=50, help="# of episodes to train")
    parser.add_argument("--meta_val_num", type=int, default=50, help="# of episodes for val")
    parser.add_argument("--meta_test_num", type=int, default=50, help="# of episodes for test")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--gamma", type=float, default=0.5, help="loss weight coefficient")
    parser.add_argument("--n_layers", type=int, default=2, help="# of EGNN layers")
    parser.add_argument("--anchor_size", type=int, default=16, help="# of virtual anchor nodes")
    parser.add_argument("--out_dir", type=str, default=None, help="dir to write run.json / episodes.jsonl (None: no files)")
    parser.add_argument("--mem", action="store_true", help="enable relation memory (record / diagnose)")
    parser.add_argument("--mem_k", type=int, default=10, help="# of retrieved memory entries")
    parser.add_argument("--mem_tau", type=float, default=0.1, help="softmax temperature for retrieval weights")
    parser.add_argument("--mem_chunk", type=int, default=256, help="# of query edges per retrieval chunk")
    parser.add_argument("--mem_keys", type=str, default="raw", help="comma-separated memory key kinds: raw,center,str")
    parser.add_argument("--mem_arms", type=str, default="all", choices=["all", "none"], help="evaluate correction arms A/B")
    parser.add_argument("--dump_edges", action="store_true", help="dump per-edge test diagnostics to out_dir/edges_test.npz")
    parser.add_argument("--dump_emb", action="store_true", help="dump full-graph GCN embedding at the best-valid epoch to out_dir/emb_best.npy")
    parser.add_argument("--dump_test_eps", action="store_true", help="dump sampled test episodes to out_dir/test_eps.npz")
    parser.add_argument("--dump_logits", action="store_true", help="dump valid/test query x class scores used by accuracy() to out_dir/eval_logits.npz")
    parser.add_argument("--pres_lambda", type=float, default=0.0, help="T07 preservation loss weight (0: original)")
    parser.add_argument("--pres_tau", type=float, default=0.1, help="T07 preservation softmax temperature")
    parser.add_argument("--pres_m", type=int, default=1024, help="T07 preservation batch size")
    parser.add_argument("--pres_pool", type=str, default="nb", choices=["nb", "all"], help="T07 node pool: nb = non-base nodes, all = all nodes")
    parser.add_argument("--pres_teacher_hops", type=int, default=2, choices=[0, 1, 2], help="T09 teacher: row-normalized A_hat^k X")
    parser.add_argument("--pres_loss", type=str, default="kl", choices=["kl", "mse", "infonce"], help="T09 auxiliary loss form")
    parser.add_argument("--fixed_eval", action="store_true", help="T07 fixed-episode evaluation of the selected checkpoint")
    parser.add_argument("--gcn_layers", type=int, default=1, choices=[1, 2], help="1: original GCN; 2: two-layer variant (T06b)")
    parser.add_argument("--neg_sampler", type=str, default="uniform", choices=["uniform", "emb_rwr", "bsc_rwr"], help="T14 infonce batch sampler")
    parser.add_argument("--rwr_alpha", type=float, default=0.1, help="T14 RWR restart probability")
    parser.add_argument("--rwr_k", type=int, default=10, help="T14 # of neighbours in the RWR k-NN graph")
    parser.add_argument("--referee_k", type=int, default=0, help="T14 referee: mask negatives among A_hat^2 X top-k neighbours (0: off)")

    return parser.parse_known_args()


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ""
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in [
            "lr",
            "epochs",
            "device",
            "seed",
            "data_dir" "num_seed",
            "summary",
            "dropout",
            "episodes",
            "meta_val_num",
            "meta_test_num",
            "optim",
            "eps",
            "l1",
            "l2",
            "final_result",
            "n_layers",
            "anchor_size",
            "out_dir",
            "mem",
            "mem_k",
            "mem_tau",
            "mem_chunk",
            "mem_keys",
            "mem_arms",
            "dump_edges",
            "dump_emb",
            "dump_test_eps",
            "gcn_layers",
            "dump_logits",
            "pres_lambda",
            "pres_tau",
            "pres_m",
            "pres_pool",
            "fixed_eval",
            "pres_teacher_hops",
            "pres_loss",
            "neg_sampler",
            "rwr_alpha",
            "rwr_k",
            "referee_k",
        ]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]
