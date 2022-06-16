'''Code for various drug property prediction models'''
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

from modules.graph_nn import ResGatedConv


class HeadModel(nn.Module):
    def __init__(self, hid_ds):
        super().__init__()

        self.layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(len(hid_ds)-1):
            self.layers.append(
                gnn.ResGatedGraphConv(hid_ds[i], hid_ds[i+1]))
            self.batch_norm.append(
                nn.BatchNorm1d(hid_ds[i+1]))

        self.out_fc = nn.Linear(hid_ds[-1], 1)

    def forward(self, x, edge_index, batch):
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = self.batch_norm[i](x)
        # Compresses node features into single feature vector
        x = gnn.global_add_pool(x, batch)
        x = self.out_fc(x)
        return x


class RootModel(torch.nn.Module):
    def __init__(self, hid_ds):
        super(RootModel, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(len(hid_ds)-1):
            self.layers.append(
                gnn.ResGatedGraphConv(hid_ds[i], hid_ds[i+1]))
            self.batch_norm.append(
                nn.BatchNorm1d(hid_ds[i+1]))

    def forward(self, x, edge_index, batch):
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = self.batch_norm[i](x)
        # Compresses node features into single feature vector
        return x