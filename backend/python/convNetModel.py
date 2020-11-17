import torch
import torch.nn as nn

# (W - F *2P)/S + 1
# (128 - 5)/1 + 1 = 6xx
# (376 - 2)/2 + 1 = 6xx
# (188 - 3) + 1 = 16xx
# (186 - 2)/2 + 1 = 16xx
# (93 + 3) + 1 = 16xx
# (91 - 2)/2 + 1 = 16x46x46
#


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.l1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.l2 = nn.Conv2d(6, 16, 3)
        self.l3 = nn.Conv2d(6, 16, 3)
