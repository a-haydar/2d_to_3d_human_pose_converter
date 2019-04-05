import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 30)
        self.fc1_bn = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30, 20)
        self.fc2_bn = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 8)

    def forward(self, x):
        x = x.view(-1, 16)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
