'''Code for various drug property prediction models'''
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

class RootModel(nn.Module):
    '''
    Backbone graph processor to be used
    in conjunction with 1-n output heads 

                         --> HeadModel
                         |
    Inputs -> RootModel --> HeadModel
                         |
                         --> HeadModel
    '''
    def __init__(
        self, hid_ds, edge_in_d, n_lstms):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hid_ds)-1):
            self.layers.append(
                gnn.ResGatedGraphConv(
                    hid_ds[i],
                    hid_ds[i+1],
                ))
        self.out_layer = nn.Sequential(
            gnn.Set2Set(hid_ds[-1], 3, n_lstms),
            nn.Linear(hid_ds[-1]*2, hid_ds[-1]))

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.out_layer(x)
        return x
        

class HeadModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self, in_d, out_d):
        super(GATNet, self).__init__()
        self.conv1 = gnn.ResGatedGraphConv(in_d, 1024, heads=1) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = gnn.ResGatedGraphConv(1024, 512, heads=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = gnn.ResGatedGraphConv(512, 256, heads=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = gnn.ResGatedGraphConv(256, 516, heads=1)
        self.bn4 = nn.BatchNorm1d(516)
        self.conv5 = gnn.ResGatedGraphConv(516, 1024, heads=1)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = gnn.global_add_pool(x, batch)

        return x


class GCNNet(torch.nn.Module):
    def __init__(self, in_d, out_d):
        super(GCNNet, self).__init__()
        self.conv1 = gnn.ResGatedGraphConv(in_d, 512) # if you defined cache=True, the shape of batch must be same!
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = gnn.ResGatedGraphConv(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = gnn.ResGatedGraphConv(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = gnn.ResGatedGraphConv(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = gnn.ResGatedGraphConv(128, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, out_d)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = gnn.global_add_pool(x, batch)

        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.fc4(x)
        return x