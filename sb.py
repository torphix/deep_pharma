import pandas as pd
import numpy as np
import time
from utils import MolecularVocab,  numerate_features, smiles_to_graph 
from tdc.single_pred import ADME, Tox
import rdkit.Chem as Chem
from ADMET.utils import TASKS
import torch

import torch.nn as nn


class ResGCN(nn.Module):
    def __init__(self, in_d, out_d, regularize=False):
        super(ResGCN, self).__init__()
        self.regularize = regularize

        self.fc = nn.Linear(in_d, out_d)
        self.activation = nn.LeakyReLU(0.2)

        self.regularization = nn.Sequential(
            nn.BatchNorm1d(out_d),
            nn.Dropout(0.2))

    def forward(self, node_features, edge_mapping, batch_lens):
        '''
        node_features: [1, BS*L, N]
        edge_mapping: 
            - col 1 is # of edge connections
            - col 2 is nodes connected to the first
        '''
        # Embed node features
        node_features = self.fc(node_features)
        residual = node_features
        # Perform message passing 
        node_features = node_features.transpose(2,1).squeeze(0)
        node_idxs = torch.index_select(node_features, index=edge_mapping[1], dim=0)
        x = torch.zeros_like(node_features)
        new_node_features = x.index_add_(dim=0, index=edge_mapping[0], source=node_idxs)
        new_node_features = new_node_features.transpose(0,1)
        # Normalization
        node_features = (new_node_features + residual) / batch_lens
        # Regularization
        if self.regularize:
            node_features = self.regularization(node_features)
        return node_features

def data2batch(nodes:list, adj_matrices:list, add_self_connections=True):
    '''
    Converts a list of nodes and adj matrices 
    into into a single graph disjointed strucutre 
    for parrallel processing
    Note first column is the # of edge connections 
    and second column is the nodes connected to the first
    ie: [0,0,1,2] [2,3,1,1] = Node 0 is connected to 2 & 3
    node 1 is connected to node 1 & node 2 is connected to node 1
    Second column is used to select the features from node tensor
    and first column is used to add the selected nodes features
    '''
    # Edge processing
    edge_idxs = []
    for i in range(len(nodes)):
        node = nodes[i]
        n_nodes = node.shape[1]
        adj_matrix = adj_matrices[i]
        if add_self_connections:
            self_connections = torch.eye(adj_matrix.shape[-1])
            adj_matrix = adj_matrix + self_connections
        # Extract edge idxs from adj_matrix
        edges = adj_matrix.nonzero()
        edge_connection = torch.tensor(edges[:,0] * n_nodes + edges[:,1], dtype=torch.int32)
        node_idxs = torch.tensor(edges[:,0] * n_nodes + edges[:,2], dtype=torch.int32)
        edge_idx = torch.stack((edge_connection, node_idxs))
        if i != 0:
            edge_idx += n_prev_nodes
        edge_idxs.append(edge_idx)
        n_prev_nodes = n_nodes
    return torch.cat(nodes, dim=1).float(), torch.cat(edge_idxs, dim=1).int()


# Finish vectorising 

gcn=ResGCN(2, 10)
node_features_1 = torch.tensor([[
    [2,4],
    [3,3],
    [1,2],
    [3,2],
]])
node_features_2 = torch.tensor([[
    [2,4],
    [3,3],
    [1,2],
    [1,2],
    [3,2],
]])
adj_matrix_1 = torch.tensor([[
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
]])
adj_matrix_2 = torch.tensor([[
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
]])


nodes, edges_mapping = data2batch([node_features_1, node_features_2], [adj_matrix_1, adj_matrix_2])
out = gcn(nodes, edges_mapping, len(nodes))

print(out)