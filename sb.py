import pandas as pd
import numpy as np
import time
from utils import MolecularVocab,  numerate_features, smiles_to_graph 
from tdc.single_pred import ADME, Tox
import rdkit.Chem as Chem
from ADMET.utils import TASKS
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data


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
        x = gnn.global_add_pool(x, batch)
        return x

model = RootModel([10, 256, 256, 128])


vocab = MolecularVocab()
graph_1 = smiles_to_graph('CCCCCC=O', vocab.atom_stoi)
# nodes_1, edge_atrs_1, adj_matrix_1 = numerate_features(graph_1)
nodes, edges = numerate_features(graph_1)

data = Data(nodes, edges)

gnn.GCN
# print(data.x, data.edge_index.T)


# Inputs Feature vector
# Sparse fully connected 

class ResGatedConv(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W1 = nn.Linear(in_d, out_d)
        self.W2 = nn.Linear(in_d, out_d)
        self.W3 = nn.Linear(in_d, out_d)
        self.W4 = nn.Linear(in_d, out_d)

    def forward(self, x, edge_idx, add_self=True):
        '''
        x: [BS*# of nodes, N]
        edge_idx: [2, # of connections]
        add_self: if to add self features
        '''
        # Embed & Message passing
        x1 = self.W1(x)
        x2 = self.message_passing(self.W2(x), edge_idx, add_self)
        x3 = self.message_passing(self.W3(x), edge_idx, add_self)
        x4 = self.message_passing(self.W4(x), edge_idx, add_self)

        # Gate & Hadamard
        xN = torch.sigmoid(x3 + x4) * x2
        # Residual
        out = xN + x1
        return out

    def message_passing(self, x, edge_idx, add_self=True):
        if add_self:
            x = x + x
        # Select start nodes
        src = x.index_select(0, edges[0])
        tgt = torch.zeros_like(x)
        # Broad cast indices to include all features
        edge_idx = edges[1].unsqueeze(1).expand(-1, src.shape[1])
        # Add node features
        out = tgt.scatter_add(0, edge_idx, src)
        return out

x = torch.ones((5,10)).float()
edges = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0],
        [1, 0, 2, 1, 3, 2, 0]
    ])



# conv = ResGatedConv(10, 3)
# out = conv(x, edges)
# print(out)

# # 1. Select nodes using first column (note can select the same node multiple times)
# # 2. Second column of edge index serves as the scatter_add index selection
# # 3. scatter add the selected nodes from 1 using selected index from 2. -> output
# from tdc.utils import retrieve_dataset_names

# tox_datasets = retrieve_dataset_names('Tox')
# adme_datasets = retrieve_dataset_names('ADME')
# print(tox_datasets, adme_datasets)
# from tdc.utils import retrieve_label_name_list

# label_list = retrieve_label_name_list('Toxcast')
# print(label_list)
from rdkit import Chem
ATOMS = set()
def get_atoms(smiles):
    global ATOMS
    atoms = set()
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atoms.add(atom.GetSymbol())

    ATOMS = ATOMS | atoms

