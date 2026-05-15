import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool



class DrugGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=512, out_channels=64, heads=8, concat=True, edge_dim=7 )
        self.dropout1 = nn.Dropout(0.4)
        self.conv2 = GATv2Conv(in_channels=512, out_channels=64, heads=8, concat=True, edge_dim=7)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x =self.dropout1(x)
        x = self.conv2(x, edge_index, edge_attr)
        return global_mean_pool(x, data.batch)


class TargetGNN_0(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1
        )
        self.conv2 = GATv2Conv(
            in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return global_mean_pool(x, data.batch)


class TargetGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1)
        self.dropout1 = nn.Dropout(0.4)
        self.conv2 = GATv2Conv(in_channels=2560, out_channels=320, heads=8, concat=True, edge_dim=1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_attr)

        out = global_mean_pool(x, data.batch)
        return out


class DeepMG(nn.Module):

    def __init__(self):
        super().__init__()
        self.drug_0 = DrugGNN()
        self.target_0 = TargetGNN()
        self.target_1 = TargetGNN()
        self.out = nn.Sequential(
            nn.Linear(512 + 2560 + 2560, 2560),
            nn.BatchNorm1d(2560),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
            nn.Linear(2560, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
            nn.Linear(1280, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
            nn.Linear(320, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_graph_0, target_graph_0, target_graph_1):

        combined = torch.cat((self.drug_0(drug_graph_0), self.target_0(target_graph_0), self.target_1(target_graph_1)), dim=1)
        return self.out(combined)
