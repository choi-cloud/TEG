import torch
import torch.nn.functional as F


class RelationMemory:
    """Relation memory over directed episode edges (r <- c).

    Stores global node ids and the base-label same-class flag during training,
    and rebuilds keys from the current model's first-layer EGNN message at evaluation time.
    """

    def __init__(self, k, tau, chunk, device):
        self.k = k
        self.tau = tau
        self.chunk = chunk
        self.device = device
        self.pairs = {}  # (r_gid, c_gid) -> same (0/1), insertion-ordered
        self.keys = None  # (M, d) L2-normalized messages
        self.msgs = None  # (M, d) raw messages (used by arm A)
        self.same = None  # (M,) float

    def __len__(self):
        return len(self.pairs)

    def add(self, r_gid, c_gid, same):
        key = (int(r_gid), int(c_gid))
        if key not in self.pairs:
            self.pairs[key] = int(same)

    @staticmethod
    def edge_messages(egnn, x, str_feat, r, c):
        # first-layer message exactly as EGCL.forward computes it: coord2dist -> msg_model
        coord_diff = x[r] - x[c]
        sqr_dist = torch.sum(coord_diff**2, 1).unsqueeze(1)
        msg = egnn.gcl_0.msg_model(str_feat[r], str_feat[c], sqr_dist)
        return msg, sqr_dist.squeeze(1)

    def node_coords(self, conv, egnn, features, edges):
        emb = conv(features, edges)
        return egnn.LayerNorm(emb)

    def build(self, conv, egnn, features, edges, str_feat, key_kinds=("raw",)):
        was_training = (conv.training, egnn.training)
        conv.eval()
        egnn.eval()
        with torch.no_grad():
            x = self.node_coords(conv, egnn, features, edges)
            rc = torch.tensor(list(self.pairs.keys()), dtype=torch.long, device=self.device)
            msgs = []
            for s in range(0, rc.shape[0], 65536):
                m, _ = self.edge_messages(egnn, x, str_feat, rc[s : s + 65536, 0], rc[s : s + 65536, 1])
                msgs.append(m)
            self.msgs = torch.cat(msgs)
            self.keys = F.normalize(self.msgs, dim=1)
            self.same = torch.tensor(list(self.pairs.values()), dtype=torch.float, device=self.device)

            # extra key kinds (T05): mean-centered message, mean-centered structural pair [s_r, s_c]
            self.key_mats = {"raw": self.keys}
            if "center" in key_kinds:
                self.mu_m = self.msgs.mean(0, keepdim=True)
                self.key_mats["center"] = F.normalize(self.msgs - self.mu_m, dim=1)
            if "str" in key_kinds:
                s_cat = torch.cat([str_feat[rc[:, 0]], str_feat[rc[:, 1]]], dim=1)
                self.mu_s = s_cat.mean(0, keepdim=True)
                self.key_mats["str"] = F.normalize(s_cat - self.mu_s, dim=1)
        conv.train(was_training[0])
        egnn.train(was_training[1])
        return x

    def query(self, m_edges, key="raw", s_edges=None):
        # key "raw": T03a rule; "center": (m - mu_m); "str": ([s_r, s_c] - mu_s), all L2-normalized
        with torch.no_grad():
            if key == "raw":
                q = F.normalize(m_edges, dim=1)
            elif key == "center":
                q = F.normalize(m_edges - self.mu_m, dim=1)
            else:
                q = F.normalize(s_edges - self.mu_s, dim=1)
            keys = self.key_mats[key]
            k = min(self.k, keys.shape[0])
            sims, idxs = [], []
            for s in range(0, q.shape[0], self.chunk):
                sim = q[s : s + self.chunk] @ keys.T
                top_sim, top_idx = torch.topk(sim, k, dim=1)
                sims.append(top_sim)
                idxs.append(top_idx)
            top_sim = torch.cat(sims)
            idx = torch.cat(idxs)
            w = torch.softmax(top_sim / self.tau, dim=1)
            p_hat = (w * self.same[idx]).sum(1)
            return {"p_hat": p_hat, "top1_sim": top_sim[:, 0], "topk_last_sim": top_sim[:, -1], "idx": idx, "w": w}
