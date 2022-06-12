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

def data2batch(nodes:list, adj_matrices:list, add_self_connections=True):
    '''
    Converts a list of nodes and adj matrices 
    into into a single graph disjointed strucutre 
    for parrallel processing
    '''
    for i in range(len(nodes)):
        node = nodes[i]
        n_nodes = node.shape[0]
        adj_matrix = adj_matrices[i]
        if add_self_connections:
            self_connections = torch.eye(adj_matrix.shape[-1])
            adj_matrix = adj_matrix + self_connections
        # Extract edge idxs from adj_matrix
        edges = adj_matrix.nonzero()
        edge_connection = edges[:,0] * n_nodes + edges[:,1]
        node_idxs = edges[:,0] * n_nodes + edges[:,2]
    return edge_connection, node_idxs


# Finish vectorising 

gcn=GCN(1, 2)
node_features = torch.tensor([[
    [2,4],
    [3,3],
    [1,2],
    [3,2],
]])
adj_matrix = torch.tensor([[
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
]])

import networkx as nx
vocab = MolecularVocab()
graph_1 = smiles_to_graph('CCCC', vocab.atom_stoi)
nodes_1, edge_atrs_1, adj_matrix_1 = numerate_features(graph_1)
print(nx.to_scipy_sparse_matrix(graph_1))

edge_connection, node_idxs = data2batch([node_features], [adj_matrix])
print(edge_connection, node_idxs)

gcn.fc.weight.data = torch.tensor(torch.eye(4))
gcn.fc.bias.data = torch.tensor(torch.tensor([0,0,0,0]))

out = gcn(node_features.transpose(2,1).float(), adj_matrix.float(), True)
print(out.T)

# x = torch.tensor([
#         [2., 4.],
#         [1., 2.],
#         [3., 2.],
#         [3., 3.],
#         [1., 2.],
#         [3., 2.],
#         [2., 4.],
#         [3., 3.],
#         [1., 2.],
#         [3., 2.],
#         [2., 4.],
#         [3., 3.],
#         [1., 2.],
#         [3., 2.]])

# y = torch.zeros_like(x).float()
# x = x.float().index_add(0, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), x.float())
# print(x)

# z = torch.zeros_like(x)
# z = z.scatter_add_(0, torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]]), x.float())
# print(z)

# Weight the values in the adjacency matrix rather than naively applying, this is because 
# If a node has the same connection (including self connections) as another node then it will return the same value thereby removing the unique representation of that nodes ie: nodes are now identical
# Implent the GAT

# Select index node features
# Summate node features selected features
# Normalise