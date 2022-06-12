import pandas as pd
import numpy as np
import time
from utils import MolecularVocab,  numerate_features, smiles_to_graph 
from tdc.single_pred import ADME, Tox
import rdkit.Chem as Chem
from ADMET.utils import TASKS
import torch

import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_d, out_d, regularize=False):
        super(GCN, self).__init__()
        self.regularize = regularize

        self.fc = nn.Linear(in_d, out_d)
        self.activation = nn.LeakyReLU(0.2)

        self.regularization = nn.Sequential(
            nn.BatchNorm1d(out_d),
            nn.Dropout(0.2))

    def add_self_loops(self, adj_matrix):
        '''
        As the adj_matrix is a sparse matrix ie:
        mirrored along the identity axis must add 
        self loops to both rows and columns
        '''

        return adj_matrix
        

    def forward(self, node_features, edge_mapping):
        '''
        node_features: [BS, L, N]
        adj_matrix: [BS, LxL] should be a sparse adj_matrix
        ie: 1 = edge link 0 = no edge link
        degree is the # of nodes connected to another node
        '''
        degree = adj_matrix.sum(dim=2)
        n_nodes = node_features.shape[2]
        # Embed node features
        node_features = self.fc(node_features)
        residual = node_features
        # Perform message passing 
        # Select connected nodes
        edges = adj_matrix.nonzero()
        # Efficient selection of connected nodes
        edge_connection = edges[:,0] * n_nodes + edges[:,1]
        node_idxs = edges[:,0] * n_nodes + edges[:,2]
        node_features = node_features.transpose(2,1).squeeze(0)
        node_idxs = torch.index_select(node_features, index=node_idxs, dim=0)
        # Message pass to connected nodes
        x = torch.zeros_like(node_features)
        node_features = x.index_add_(dim=0, index=edge_connection, source=node_idxs)
        # Trim to size
        node_features = node_features[:n_nodes, :].squeeze(0)
        # Normalization
        node_features = (node_features + residual) / degree
        # # Regularization
        # if self.regularize:
        #     node_features = self.regularization(node_features)
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
        edge_connection = edges[:,0] * n_nodes + edges[:,1]
        node_idxs = edges[:,0] * n_nodes + edges[:,2]
        edge_idx = torch.stack((edge_connection, node_idxs))
        if i != 0:
            print(n_prev_nodes)
            edge_idx += n_prev_nodes
        edge_idxs.append(edge_idx)
        n_prev_nodes = n_nodes
    return torch.cat(edge_idxs, dim=1)


# Finish vectorising 

gcn=GCN(1, 2)
node_features = torch.tensor([[
    [2,4],
    [3,3],
    [1,2],
    [3,2],
]])
adj_matrix = torch.tensor([[
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
]])


import networkx as nx
vocab = MolecularVocab()
graph_1 = smiles_to_graph('CCCC', vocab.atom_stoi)
nodes_1, adj_matrix_1, edge_attr = numerate_features(graph_1)
sparse = nx.to_scipy_sparse_matrix(graph_1)

edges_mapping = data2batch([node_features, node_features], [adj_matrix, adj_matrix])
print(edges_mapping)

gcn.fc.weight.data = torch.tensor(torch.eye(4))
gcn.fc.bias.data = torch.tensor(torch.tensor([0,0,0,0]))

out = gcn(node_features.transpose(2,1).float(), adj_matrix.float(), True)
print(out.T)