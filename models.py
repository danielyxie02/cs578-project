import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc3 = nn.Linear(16 * 5 * 5, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.fc3(x.view(-1, 16 * 5 * 5)))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x