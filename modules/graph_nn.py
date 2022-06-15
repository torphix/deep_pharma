import torch
import torch.nn as nn

from utils import MolecularVocab, numerate_features, smiles_to_graph


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
        

    def forward(self, node_features, adj_matrix, add_self_loop=True):
        '''
        node_features: [BS, L, N]
        adj_matrix: [BS, LxL] should be a sparse adj_matrix
        ie: 1 = edge link 0 = no edge link
        degree is the # of nodes connected to another node
        '''
        if add_self_loop:
            adj_matrix = self.add_self_loops(adj_matrix)
            
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
        print(node_idxs, 'node_idxs')
        nodes = torch.index_select(node_features, index=node_idxs, dim=0)
        # Message pass to connected nodes
        x = torch.zeros_like(node_features)
        node_features = x.index_add_(dim=0, index=edge_connection, source=nodes)
        # Trim to size
        node_features = node_features[:n_nodes, :].squeeze(0)
        # Normalization
        node_features = (node_features + residual) / degree
        # # Regularization
        # if self.regularize:
        #     node_features = self.regularization(node_features)
        return node_features


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

    def message_passing(self, x, edges, add_self=True):
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