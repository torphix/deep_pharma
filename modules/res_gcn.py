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