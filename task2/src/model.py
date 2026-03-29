import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool


class GNNClassifier(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=2):
        super().__init__()

        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), aggr='max')

        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2)
        ), aggr='max')

        self.conv3 = EdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_channels * 2, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels * 4)
        ), aggr='max')

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x