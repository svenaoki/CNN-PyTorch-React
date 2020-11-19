import torch
import torch.nn as nn
import torch.nn.functional as F
# (W - F *2P)/S + 1
# l1 (3,6,5) + pool
# (128 - 5)/1 + 1 = 6x124x124
# (124 - 2)/2 + 1 = 6x62x62

# l2 (6,16,3) + pool
# (62 - 3) + 1 = 16x60x60
# (60 - 2)/2 + 1 = 16x30x30

# l3 (16,6,3) + pool
# (29 - 3) + 1 = 6x27x27
# (27 - 2)/2 + 1 = 6x14x14


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.l1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.l2 = nn.Conv2d(6, 16, 3)
        self.l3 = nn.Conv2d(16, 6, 3)
        self.f1 = nn.Linear(6*14*14, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.l1(x)))
        x = self.pool(F.relu(self.l2(x)))
        x = self.pool(F.relu(self.l3(x)))
        x = x.view(-1, 6*14*14)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x
