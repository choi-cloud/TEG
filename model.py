from calendar import c
from layers.GCN import GCN, GCN2
from layers.EGNN import EGNN
from embedder import embedder
from tqdm.auto import tqdm
from utils import *
from torch import optim
import torch.nn.functional as F
import torch
import os
import json
import time
from sklearn.metrics import roc_auc_score
from memory import RelationMemory
from argument import config2string, parse_args

MEM_ARMS = [("A_0.1", "A", 0.1), ("A_0.3", "A", 0.3), ("A_0.5", "A", 0.5), ("B_0.1", "B", 0.1), ("B_0.3", "B", 0.3), ("B_0.5", "B", 0.5)]
MEM_EP_KEYS = ["n_edges", "auc_phat", "auc_dist", "mean_phat_same", "mean_phat_diff", "mean_top1_sim"]


class teg_trainer(embedder):
    def __init__(self, args, conf, set_seed):
        embedder.__init__(self, args, conf, set_seed)
        if args.gcn_layers == 2:
            # T06b variant model; --gcn_layers 1 (default) keeps the original line below
            self.conv = GCN2(self.features.shape[1], 64, conf["gcn_out"], args.dropout).to(self.device)
        else:
            self.conv = GCN(self.features.shape[1], conf["gcn_out"], args.dropout).to(self.device)
        self.egnn = EGNN(self.structural_features.shape[1], conf["egnn_in"], n_layers=args.n_layers).to(self.device)

        self.optim = optim.Adam([{"params": self.conv.parameters()}, {"params": self.egnn.parameters()}], lr=args.lr, weight_decay=5e-4)

        self.config_str = config2string(args)
        self.set_seed = set_seed

        if args.mem:
            self.memory = RelationMemory(args.mem_k, args.mem_tau, args.mem_chunk, self.device)
            self.labels_np = self.labels.cpu().numpy()
            self.mem_keys = args.mem_keys.split(",")
            assert set(self.mem_keys) <= {"raw", "center", "str"}, args.mem_keys
            self.mem_arms = MEM_ARMS if args.mem_arms == "all" else []

    def mem_record(self, id_support, id_query):
        # (q <- s) and (s <- q) for every query-support pair; base labels only
        for q in id_query:
            for s in id_support:
                same = int(self.labels_np[q] == self.labels_np[s])
                self.memory.add(q, s, same)
                self.memory.add(s, q, same)

    def mem_diag(self, embeds_epi, epi_str, edge_index, gids):
        # first-layer messages of this episode, computed as in EGNN.forward (LayerNorm -> coord2dist -> msg_model)
        row, col = edge_index
        x = self.egnn.LayerNorm(embeds_epi)
        sqr_dist, _ = self.egnn.gcl_0.coord2dist(edge_index, x)
        msg = self.egnn.gcl_0.msg_model(epi_str[row], epi_str[col], sqr_dist)
        res = self.memory.query(msg)
        lab = self.labels_np[gids]
        same = (lab[row.cpu().numpy()] == lab[col.cpu().numpy()]).astype(int)
        res["msg"] = msg
        res["same"] = same
        res["neg_dist"] = (-sqr_dist.squeeze(1)).cpu().numpy()
        s_edges = torch.cat([epi_str[row], epi_str[col]], dim=1)
        res["by_key"] = {key: (res if key == "raw" else self.memory.query(msg, key, s_edges)) for key in self.mem_keys}
        return res

    @staticmethod
    def mem_stats(p_hat, neg_dist, same, top1_sim):
        both = 0 < same.sum() < len(same)
        return {
            "n_edges": int(len(same)),
            "auc_phat": float(roc_auc_score(same, p_hat)) if both else None,
            "auc_dist": float(roc_auc_score(same, neg_dist)) if both else None,
            "mean_phat_same": float(p_hat[same == 1].mean()) if (same == 1).any() else None,
            "mean_phat_diff": float(p_hat[same == 0].mean()) if (same == 0).any() else None,
            "mean_top1_sim": float(top1_sim.mean()),
        }

    def episode_forward(self, epi_str, embeds_epi, edge_index, n_support, msg_fn=None, w_fn=None):
        # ______________________
        # EGNN - Task adaptation
        epi_str, embeds_epi = self.egnn(epi_str, embeds_epi, edge_index, msg_fn, w_fn)
        self.last_final_coords = embeds_epi  # read-only tap for d_final (T05)

        # __________
        # Prototypes
        embeds_spt = embeds_epi[:n_support, :]
        embeds_qry = embeds_epi[n_support:, :]

        embeds_spt = embeds_spt.view([self.n_way, self.k_shot, embeds_spt.shape[1]])

        embeds_proto = embeds_spt.mean(1)

        # __________
        # Prediction
        dists_output = euclidean_dist(embeds_qry, embeds_proto)
        output = F.log_softmax(-dists_output, dim=1)
        output_softmax = F.softmax(-dists_output, dim=1)

        return output, output_softmax

    def arm_fns(self, name, kind, beta, mem_res):
        if kind == "A":
            # v_bar: retrieval-weighted mean of stored raw memory messages
            v_bar = (mem_res["w"].unsqueeze(2) * self.memory.msgs[mem_res["idx"]]).sum(1)
            return (lambda m: (1 - beta) * m + beta * v_bar), None

        p_hat = mem_res["p_hat"].unsqueeze(1)

        def w_fn(w):
            sigma = w.std()
            if sigma.item() < 1e-8:
                self.sigma_skip[name] += 1
                return w
            return w - beta * sigma * (2 * p_hat - 1)

        return None, w_fn

    def arm_beta0_check(self, epi_str, embeds_epi, edge_index, n_support, mem_res, off_softmax, label_list, off_acc):
        out = {"off_acc": float(off_acc)}
        for kind in ("A", "B"):
            msg_fn, w_fn = self.arm_fns(f"{kind}_0.0", kind, 0.0, mem_res)
            _, sm = self.episode_forward(epi_str, embeds_epi, edge_index, n_support, msg_fn, w_fn)
            out[f"{kind}_0.0_acc"] = float(accuracy(sm.cpu().detach(), label_list))
            out[f"{kind}_0.0_max_abs_diff_softmax"] = float((sm - off_softmax).abs().max())
        return out

    def dump_embedding(self, epoch):
        # GCN output before EGNN/LayerNorm, eval mode, no_grad; overwrites out_dir/emb_best.npy
        self.conv.eval()
        with torch.no_grad():
            emb = self.conv(self.features, self.edges)
        os.makedirs(self.args.out_dir, exist_ok=True)
        np.save(os.path.join(self.args.out_dir, "emb_best.npy"), emb.cpu().numpy().astype(np.float32))
        self.emb_epoch = epoch
        if not os.path.exists(os.path.join(self.args.out_dir, "split.json")):
            to_py = lambda xs: [x.item() if hasattr(x, "item") else x for x in xs]
            with open(os.path.join(self.args.out_dir, "split.json"), "w") as f:
                json.dump(
                    {
                        "class_list_train": to_py(self.class_list_train),
                        "class_list_valid": to_py(self.class_list_valid),
                        "class_list_test": to_py(self.class_list_test),
                    },
                    f,
                )
            np.save(os.path.join(self.args.out_dir, "labels.npy"), self.labels.cpu().numpy())

    def mem_configs(self):
        # TEG rule per config: best valid by strict '<', test taken whenever valid equals the best (later epoch wins ties)
        configs = {}
        for name in ["off"] + [a[0] for a in self.mem_arms]:
            best_valid, best_epoch, test_at_best = 0, 0, None
            for rec in self.epoch_records:
                v, t = rec["arm_valid_acc"][name], rec["arm_test_acc"][name]
                if best_valid < v:
                    best_valid, best_epoch = v, rec["epoch"]
                if v == best_valid:
                    test_at_best = t
            configs[name] = {"best_valid_epoch": best_epoch, "best_valid_acc": best_valid, "test_acc_at_best_valid": test_at_best}
        return configs

    def mem_sanity_check(self):
        # one training episode of epoch 1 (its pairs are in memory), queried right after build
        id_support, id_query, ep_idx = self.sanity_ids
        n_s, n_q = len(id_support), len(id_query)
        edge1 = torch.LongTensor([i for i in range(n_s, n_s + n_q) for _ in range(n_s)])
        edge2 = torch.LongTensor(list(range(n_s)) * n_q)
        edge_index = torch.stack((torch.cat([edge1, edge2]), torch.cat([edge2, edge1]))).to(self.device)
        self.conv.eval()
        self.egnn.eval()
        emb = self.conv(self.features, self.edges)
        gids = np.concatenate([id_support, id_query])
        res = self.mem_diag(emb[gids], self.structural_features[gids], edge_index, gids)
        stats = self.mem_stats(res["p_hat"].cpu().numpy(), res["neg_dist"], res["same"], res["top1_sim"].cpu().numpy())
        return {"epoch": 1, "train_ep_idx": int(ep_idx), "n_edges": stats["n_edges"], "auc_phat": stats["auc_phat"], "top1_sim_mean": stats["mean_top1_sim"]}

    def train_epoch(self, mode, n_episode, epoch):

        loss_fn = torch.nn.NLLLoss()

        if mode == "train":
            if epoch != 0:
                self.conv.train()
                self.egnn.train()
            else:
                self.conv.eval()
                self.egnn.eval()

        else:
            self.conv.eval()
            self.egnn.eval()

        if mode == "train" or mode == "valid":
            loss_epoch = 0

        acc_epoch = []
        f1_epoch = []
        self.arm_scores = {name: [] for name in ["off"] + [a[0] for a in (self.mem_arms if self.args.mem else [])]}

        for episode in range(n_episode):

            if mode == "train":
                self.optim.zero_grad()

            if mode == "train":
                class_selected = random.sample(self.class_list_train, self.args.way)

            elif mode == "valid":
                class_selected = random.sample(self.class_list_valid, self.args.way)

            elif mode == "test":
                class_selected = random.sample(self.class_list_test, self.args.way)

            id_support, id_query, class_selected = task_generator_in_class(self.id_by_class, class_selected, self.n_way, self.k_shot, self.n_query)

            if self.args.dump_test_eps and mode == "test":
                # read-only copy of the already-sampled test episode (T06b)
                self.test_eps.append((epoch, episode, np.array(id_support), np.array(id_query), [c.item() if hasattr(c, "item") else c for c in class_selected]))

            # ________________
            # graph conv (GCN)
            embeddings = self.conv(self.features, self.edges)

            # _____________
            # Task sampling
            embeds_spt = embeddings[id_support]
            embeds_qry = embeddings[id_query]

            embeds_epi = torch.cat([embeds_spt, embeds_qry])

            # structural features
            spt_str = self.structural_features[id_support]
            qry_str = self.structural_features[id_query]

            epi_str = torch.cat([spt_str, qry_str])

            # _______________________________
            # calculate a graph embedder loss
            gcn_spt = embeds_epi[: len(id_support), :]
            gcn_qry = embeds_epi[len(id_support) :, :]
            gcn_spt = gcn_spt.view([self.n_way, self.k_shot, gcn_spt.shape[1]])
            proto_embeds_gcn = gcn_spt.mean(1)
            dists_gcn = euclidean_dist(gcn_qry, proto_embeds_gcn)
            output_gcn = F.log_softmax(-dists_gcn, dim=1)

            # ___________________
            # Task-specific graph
            edge1 = []
            for i in range(len(id_support), len(id_support) + len(id_query)):
                temp = [i] * len(id_support)
                edge1.extend(temp)
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(list(range(len(id_support))) * len(id_query))
            edge1_bi = torch.cat([edge1, edge2])
            edge2_bi = torch.cat([edge2, edge1])
            edge_index = torch.stack((edge1_bi, edge2_bi)).to(self.device)

            # ______________________________________________
            # Relation memory (record / diagnose, no effect on the model)
            mem_ep = None
            if self.args.mem and epoch >= 1:
                if mode == "train":
                    self.mem_record(id_support, id_query)
                    if epoch == 1:
                        self.sanity_ids = (id_support, id_query, episode)
                else:
                    gids = np.concatenate([id_support, id_query])
                    res = self.mem_diag(embeds_epi, epi_str, edge_index, gids)
                    p_hat = res["p_hat"].cpu().numpy()
                    top1 = res["top1_sim"].cpu().numpy()
                    mem_ep = self.mem_stats(p_hat, res["neg_dist"], res["same"], top1)
                    mem_res = res
                    if mode == "test":
                        self.mem_pool["p_hat"].append(p_hat)
                        self.mem_pool["neg_dist"].append(res["neg_dist"])
                        self.mem_pool["same"].append(res["same"])

            # _____________________________________
            # EGNN -> Prototypes -> Prediction (off)
            output, output_softmax = self.episode_forward(epi_str, embeds_epi, edge_index, len(id_support))

            if self.args.mem and epoch >= 1 and mode == "test":
                # TEG final-coordinate pair distance on the task graph edges (r <- c), read from the off output
                z = self.last_final_coords
                d_final = torch.sum((z[edge_index[0]] - z[edge_index[1]]) ** 2, 1).cpu().numpy()
                self.mem_pool["d_final"].append(d_final)
                for key, kres in mem_res["by_key"].items():
                    self.mem_pool["phat_" + key].append(kres["p_hat"].cpu().numpy())
                if self.args.dump_edges:
                    n_e = len(d_final)
                    dump = {
                        "epoch": np.full(n_e, epoch, dtype=np.int16),
                        "ep_idx": np.full(n_e, episode, dtype=np.int16),
                        "dir": np.repeat(np.array([0, 1], dtype=np.int8), n_e // 2),
                        "same": mem_res["same"].astype(np.int8),
                        "d0": -mem_res["neg_dist"],
                        "d_final": d_final,
                    }
                    for key, kres in mem_res["by_key"].items():
                        dump["phat_" + key] = kres["p_hat"].cpu().numpy()
                        dump["top1_" + key] = kres["top1_sim"].cpu().numpy()
                        dump["top10_" + key] = kres["topk_last_sim"].cpu().numpy()
                    self.edge_dump.append(dump)

            # _________________________
            # Relabeling for meta-tasks
            label_list = torch.LongTensor([class_selected.index(i) for i in self.labels[id_query]]).to(self.device)

            if mode == "train" or mode == "valid":

                # ___________
                # Loss update
                loss_l1_train = loss_fn(output, label_list)  # Network Loss
                loss_l2_train = loss_fn(output_gcn, label_list)  # Graph Embedder Loss

                loss_train = self.args.gamma * loss_l1_train + (1 - self.args.gamma) * loss_l2_train

            if mode == "train":
                if epoch != 0:
                    loss_train.backward()
                    self.optim.step()
                else:
                    self.optim.zero_grad()

            # ________
            # Accuracy
            output = output_softmax.cpu().detach()
            label_list = label_list.cpu().detach()

            acc_score = accuracy(output, label_list)
            f1_score = f1(output, label_list)

            acc_epoch.append(acc_score)
            f1_epoch.append(f1_score)

            # ________________________________________________________
            # Memory arms on the same episode (no new sampling / RNG)
            if self.args.mem and (mode == "valid" or mode == "test"):
                arm_acc = {"off": acc_score}
                for name, kind, beta in self.mem_arms:
                    if epoch >= 1:
                        msg_fn, w_fn = self.arm_fns(name, kind, beta, mem_res)
                        _, arm_softmax = self.episode_forward(epi_str, embeds_epi, edge_index, len(id_support), msg_fn, w_fn)
                        arm_acc[name] = accuracy(arm_softmax.cpu().detach(), label_list)
                    else:
                        arm_acc[name] = acc_score  # epoch 0: memory is empty
                for name in arm_acc:
                    self.arm_scores[name].append(arm_acc[name])
                if epoch == 1 and mode == "test" and self.beta0_check is None and self.mem_arms:
                    self.beta0_check = self.arm_beta0_check(epi_str, embeds_epi, edge_index, len(id_support), mem_res, output_softmax, label_list, acc_score)

            if mode == "valid" or mode == "test":
                self.episode_records.append(
                    {
                        "epoch": epoch,
                        "mode": mode,
                        "ep_idx": episode,
                        "classes": [c.item() if hasattr(c, "item") else c for c in class_selected],
                        "acc": float(acc_score),
                        "f1": float(f1_score),
                    }
                )
                if self.args.mem:
                    self.episode_records[-1].update(mem_ep if mem_ep is not None else dict.fromkeys(MEM_EP_KEYS))
                    self.episode_records[-1]["arm_acc"] = {k: float(v) for k, v in arm_acc.items()}

        acc_total_epoch = sum(acc_epoch) / len(acc_epoch)
        f1_total_epoch = sum(f1_epoch) / len(f1_epoch)

        if self.args.mem and (mode == "valid" or mode == "test"):
            self.arm_epoch[mode] = {k: float(sum(v) / len(v)) for k, v in self.arm_scores.items()}

        if mode == "train":
            tqdm.write(f"acc_train : {acc_total_epoch:.4f}")

        elif mode == "valid":
            tqdm.write(f"acc_valid : {acc_total_epoch:.4f}")

        elif mode == "test":
            tqdm.write(f"acc_test : {acc_total_epoch:.4f}")

        return acc_total_epoch, f1_total_epoch

    def train(self):

        # _____________
        # Best Accuracy
        best_acc_train = 0
        best_f1_train = 0
        best_epoch_train = 0
        best_acc_valid = 0
        best_f1_valid = 0
        best_epoch_valid = 0
        best_acc_test = 0
        best_f1_test = 0
        best_epoch_test = 0

        start_time = time.time()
        self.epoch_records = []
        self.episode_records = []
        self.mem_sanity = None
        self.beta0_check = None
        self.arm_epoch = {}
        self.sigma_skip = {name: 0 for name, kind, _ in MEM_ARMS if kind == "B"}
        self.sigma_skip["B_0.0"] = 0
        self.edge_dump = []
        self.emb_epoch = None
        self.test_eps = []

        for epoch in tqdm(range(self.args.epochs + 1)):

            acc_train, f1_train = self.train_epoch("train", self.args.episodes, epoch)

            with torch.no_grad():

                if self.args.mem:
                    self.mem_pool = {"p_hat": [], "neg_dist": [], "same": [], "d_final": []}
                    self.mem_pool.update({"phat_" + key: [] for key in self.mem_keys})
                    if epoch >= 1:
                        self.memory.build(self.conv, self.egnn, self.features, self.edges, self.structural_features, self.mem_keys)
                        if epoch == 1:
                            self.mem_sanity = self.mem_sanity_check()
                            tqdm.write(f"# mem sanity : {self.mem_sanity}")

                acc_valid, f1_valid = self.train_epoch("valid", self.args.meta_val_num, epoch)

                acc_test, f1_test = self.train_epoch("test", self.args.meta_test_num, epoch)

            self.epoch_records.append(
                {
                    "epoch": epoch,
                    "train_acc": float(acc_train),
                    "train_f1": float(f1_train),
                    "valid_acc": float(acc_valid),
                    "valid_f1": float(f1_valid),
                    "test_acc": float(acc_test),
                    "test_f1": float(f1_test),
                }
            )
            if self.args.mem:
                pooled = {"mem_size": len(self.memory), "auc_phat_pooled": None, "auc_dist_pooled": None}
                pooled.update({"auc_phat_pooled_" + key: None for key in self.mem_keys} | {"auc_dfinal_pooled": None})
                if self.mem_pool["same"]:
                    same = np.concatenate(self.mem_pool["same"])
                    pooled["auc_phat_pooled"] = float(roc_auc_score(same, np.concatenate(self.mem_pool["p_hat"])))
                    pooled["auc_dist_pooled"] = float(roc_auc_score(same, np.concatenate(self.mem_pool["neg_dist"])))
                    for key in self.mem_keys:
                        pooled["auc_phat_pooled_" + key] = float(roc_auc_score(same, np.concatenate(self.mem_pool["phat_" + key])))
                    pooled["auc_dfinal_pooled"] = float(roc_auc_score(same, -np.concatenate(self.mem_pool["d_final"])))
                self.epoch_records[-1].update(pooled)
                self.epoch_records[-1]["arm_valid_acc"] = self.arm_epoch["valid"]
                self.epoch_records[-1]["arm_test_acc"] = self.arm_epoch["test"]

            if best_acc_train < acc_train:
                best_acc_train = acc_train
                best_f1_train = f1_train
                best_epoch_train = epoch

            if best_acc_valid < acc_valid:
                best_acc_valid = acc_valid
                best_f1_valid = f1_valid
                best_epoch_valid = epoch

            if best_acc_test < acc_test:
                best_acc_test = acc_test
                best_f1_test = f1_test
                best_epoch_test = epoch

            if acc_valid == best_acc_valid:
                test_acc_at_best_valid = acc_test
                test_f1_at_best_valid = f1_test

            # T06: full-graph GCN embedding at the original model-selection condition (read-only, no RNG)
            if self.args.dump_emb and self.args.out_dir is not None and acc_valid == best_acc_valid:
                self.dump_embedding(epoch)

            tqdm.write(f"# Current Settings : {self.config_str}")
            tqdm.write(f"# Best_Acc_Train : {best_acc_train:.4f}, F1 : {best_f1_train:.4f} at {best_epoch_train} epoch")
            tqdm.write(f"# Best_Acc_Valid : {best_acc_valid:.4f}, F1 : {best_f1_valid:.4f} at {best_epoch_valid} epoch")
            tqdm.write(f"# Best_Acc_Test : {best_acc_test:.4f}, F1 : {best_f1_test:.4f} at {best_epoch_test} epoch")
            tqdm.write(f"# Test_At_Best_Valid : {test_acc_at_best_valid:.4f}, F1 : {test_f1_at_best_valid:.4f} at {best_epoch_valid} epoch\n")

        np.set_printoptions(formatter={"float_kind": lambda x: "{0:0.4f}".format(x)})

        if self.args.out_dir is not None:
            final = {
                "best_acc_train": best_acc_train,
                "best_f1_train": best_f1_train,
                "best_epoch_train": best_epoch_train,
                "best_acc_valid": best_acc_valid,
                "best_f1_valid": best_f1_valid,
                "best_epoch_valid": best_epoch_valid,
                "best_acc_test": best_acc_test,
                "best_f1_test": best_f1_test,
                "best_epoch_test": best_epoch_test,
                "test_acc_at_best_valid": test_acc_at_best_valid,
                "test_f1_at_best_valid": test_f1_at_best_valid,
            }
            final = {k: (int(v) if k.startswith("best_epoch") else float(v)) for k, v in final.items()}
            os.makedirs(self.args.out_dir, exist_ok=True)
            with open(os.path.join(self.args.out_dir, "run.json"), "w") as f:
                json.dump(
                    {"config": vars(self.args), "epochs": self.epoch_records, "final": final, "wall_time_sec": time.time() - start_time}
                    | ({"emb_epoch": self.emb_epoch} if self.args.dump_emb else {})
                    | (
                        {
                            "sanity": self.mem_sanity,
                            "mem_configs": self.mem_configs(),
                            "sigma_skip": self.sigma_skip,
                            "beta0_check": self.beta0_check,
                        }
                        if self.args.mem
                        else {}
                    ),
                    f,
                    indent=2,
                )
            with open(os.path.join(self.args.out_dir, "episodes.jsonl"), "w") as f:
                for rec in self.episode_records:
                    f.write(json.dumps(rec) + "\n")
            if self.args.dump_test_eps:
                np.savez_compressed(
                    os.path.join(self.args.out_dir, "test_eps.npz"),
                    epoch=np.array([e[0] for e in self.test_eps], dtype=np.int16),
                    ep_idx=np.array([e[1] for e in self.test_eps], dtype=np.int16),
                    support=np.stack([e[2] for e in self.test_eps]),
                    query=np.stack([e[3] for e in self.test_eps]),
                    classes=np.array([e[4] for e in self.test_eps]),
                )
            if self.args.mem and self.args.dump_edges:
                np.savez_compressed(
                    os.path.join(self.args.out_dir, "edges_test.npz"),
                    **{k: np.concatenate([d[k] for d in self.edge_dump]) for k in self.edge_dump[0]},
                )

        return (
            best_acc_train,
            best_f1_train,
            best_epoch_train,
            best_acc_valid,
            best_f1_valid,
            best_epoch_valid,
            best_acc_test,
            best_f1_test,
            best_epoch_test,
            test_acc_at_best_valid,
            test_f1_at_best_valid,
        )
