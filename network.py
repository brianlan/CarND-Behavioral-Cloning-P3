import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(64, 32, 1, padding=0, stride=1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1, stride=2)

        self.fc1 = nn.Linear(20 * 20 * 32, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight)

    def forward(self, x):
        x_ = x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) + x_
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x_ = x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x)) + x_
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
