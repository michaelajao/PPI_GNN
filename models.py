# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR


class GCNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()
        print('GCNN Loaded')
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.n_output)

    def forward(self, inputs):
        (x1s, edge_index1s), (x2s, edge_index2s) = inputs
        batch_size = len(x1s)
        outputs = []
        
        for i in range(batch_size):
            # Process protein 1
            x1 = self.pro1_conv1(x1s[i], edge_index1s[i])
            x1 = self.relu(x1)
            x1 = torch.mean(x1, dim=0, keepdim=True)  # Global mean pooling
            x1 = self.relu(self.pro1_fc1(x1))
            x1 = self.dropout(x1)

            # Process protein 2
            x2 = self.pro2_conv1(x2s[i], edge_index2s[i])
            x2 = self.relu(x2)
            x2 = torch.mean(x2, dim=0, keepdim=True)  # Global mean pooling
            x2 = self.relu(self.pro2_fc1(x2))
            x2 = self.dropout(x2)

            # Combine features
            xc = torch.cat((x1, x2), 1)
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            out = self.out(xc)
            out = self.sigmoid(out)
            outputs.append(out)
        
        return torch.cat(outputs, 0)

class AttGNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, dropout=0.2, heads=1):
        super(AttGNN, self).__init__()
        print('AttGNN Loaded')
        self.hidden = 8
        self.heads = 1
        self.pro1_conv1 = GATConv(num_features_pro, self.hidden * 16, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)
        self.pro2_conv1 = GATConv(num_features_pro, self.hidden * 16, heads=self.heads, dropout=0.2)
        self.pro2_fc1 = nn.Linear(128, output_dim)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)

    def forward(self, inputs):
        (x1s, edge_index1s), (x2s, edge_index2s) = inputs
        batch_size = len(x1s)
        outputs = []
        
        for i in range(batch_size):
            # Process protein 1
            x1 = self.pro1_conv1(x1s[i], edge_index1s[i])
            x1 = self.relu(x1)
            x1 = torch.mean(x1, dim=0, keepdim=True)  # Global mean pooling
            x1 = self.relu(self.pro1_fc1(x1))
            x1 = self.dropout(x1)

            # Process protein 2
            x2 = self.pro2_conv1(x2s[i], edge_index2s[i])
            x2 = self.relu(self.pro2_fc1(x2))
            x2 = torch.mean(x2, dim=0, keepdim=True)  # Global mean pooling
            x2 = self.relu(x2)
            x2 = self.dropout(x2)

            # Combine features
            xc = torch.cat((x1, x2), 1)
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            out = self.out(xc)
            out = self.sigmoid(out)
            outputs.append(out)
        
        return torch.cat(outputs, 0)

net = GCNN()
print(net)

net_GAT = AttGNN()
print(net_GAT)

