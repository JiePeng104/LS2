import torch.nn as nn
import torch.nn.functional as F


class last_linear(nn.Module):
    def __init__(self, num_classes=10):
        super(last_linear, self).__init__()
        self.linear = nn.Linear(512 * 1, num_classes)

    def forward(self, x):
        out = self.linear(x)
        # out = self.linear(out)
        return out