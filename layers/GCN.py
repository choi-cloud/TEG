import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return x


class GCN2(torch.nn.Module):
    """2-layer variant (T06b, --gcn_layers 2): dropout -> GCNConv(in, hid) -> ReLU -> dropout -> GCNConv(hid, out)."""

    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim, cached=True, normalize=True)
        self.conv2 = GCNConv(hid_dim, out_dim, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
