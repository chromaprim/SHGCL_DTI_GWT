import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import sys
sys.path.append('../')
from src.layers.SemanticsAttention import SemanticsAttention


class WeightedSumConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class GraphWaveletTransform(nn.Module):
    def __init__(self, in_ft, out_ft, J=3, bias=True):
        super().__init__()
        self.J = J
        self.max_scale = 2 ** (J - 1)
        self.conv = WeightedSumConv()
        total_dim = in_ft * (1 + (J - 1) + (J * (J - 1) // 2))
        self.fc = nn.Linear(total_dim, out_ft, bias=False)
        self.act = nn.PReLU()
        self.bias = nn.Parameter(torch.zeros(out_ft)) if bias else None
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, seq, adj):
        device = seq.device
        edge_index = adj.coalesce().indices().to(device)
        edge_weight = adj.coalesce().values().to(device)
        diff_list = []
        x_curr = seq
        for step in range(1, self.max_scale + 1):
            x_curr = self.conv(x_curr, edge_index, edge_weight)
            if (step & (step - 1)) == 0:
                diff_list.append(x_curr)

        F0 = diff_list[-1]
        F1 = torch.cat([torch.abs(diff_list[i - 1] - diff_list[i]) for i in range(1, len(diff_list))], 1)
        F2 = self._second_order_feature(diff_list, edge_index, edge_weight)
        feats = torch.cat([F0, F1, F2], 1)

        out = self.fc(feats)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

    def _second_order_feature(self, diff_list, edge_index, edge_weight):
        U = torch.cat(diff_list, 1)
        num_feats = diff_list[0].size(1)

        U_diff = [U]
        for _ in range(1, self.max_scale):
            U_diff.append(self.conv(U_diff[-1], edge_index, edge_weight))

        features = []
        for j in range(self.J):
            for jp in range(j + 1, self.J):
                start = j * num_feats
                end = (j + 1) * num_feats
                delta = torch.abs(U_diff[jp][:, start:end] - U_diff[jp - 1][:, start:end])
                features.append(delta)
        return torch.cat(features, 1)


class MpEncoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop, J=3):
        super().__init__()
        self.P = P
        self.processors = nn.ModuleList([
            GraphWaveletTransform(hidden_dim, hidden_dim, J=J)
            for _ in range(P)
        ])
        self.attention = SemanticsAttention(hidden_dim, attn_drop)

    def forward(self, h, adj_list):
        if self.P == 0:
            return h
        embeddings = [proc(h, adj) for proc, adj in zip(self.processors, adj_list)]
        return self.attention(embeddings)


class MainModel(nn.Module):
    def __init__(self, mps_len_dict, hid_dim=64, attn_drop=0.5, J=3):
        super().__init__()
        self.mpencoder = nn.ModuleDict({
            k: MpEncoder(v, hid_dim, attn_drop, J=J)
            for k, v in mps_len_dict.items()
        })
        self.mpencoder2 = nn.ModuleDict({
            k: MpEncoder(v, hid_dim, attn_drop, J=J)
            for k, v in mps_len_dict.items()
        })

    def forward(self, x, mps_dict):
        enc_outs = [encoder(x, adj_list) for encoder, adj_list in zip(self.mpencoder.values(), mps_dict.values())]
        dec_outs = [decoder(x, adj_list) for decoder, adj_list in zip(self.mpencoder2.values(), mps_dict.values())]
        return torch.cat(enc_outs + dec_outs, dim=1)


