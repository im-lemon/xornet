import torch.nn as nn, torch

class XORnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = nn.Linear(2,4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4,1)

    def forward(self, x):
        x = self.layer1(x)
        x=self.relu(x)
        x = self.layer2(x)

        return x