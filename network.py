import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(160 * 320 * 3, 1024)
        # self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x