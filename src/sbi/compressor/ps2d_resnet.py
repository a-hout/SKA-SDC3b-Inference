"""
resnet, train based on dataset, then use the trained resnet as the embedding network
"""

import torch.nn as nn
from torchvision.models import (
    resnet18,  # smallest resnet since our images are small (10x10)
)


class PS2DResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)  # we want to train the weights ourselves
        self.resnet.conv1 = nn.Conv2d(
            in_channels=3,  # the three 3 redshift bins
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )  # small-ish kernel size and stride + padding since we only have 10x10 to work with, this is a modification of the first layer of the resnet

        self.resnet.maxpool = nn.Identity()  # this is basically a pass through because per default resnet uses max pooling but 10x10 is too small for that

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 64)
        self.end_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(),  # Gelu not needed here
            nn.Linear(64, 3),  # down to 3, for the three redshift bins
        )

    def forward(self, x):
        # shape (batch, 3, 10, 10) or entirety (all, 3, 10, 10), where to define batch size when training resnet?
        x = self.resnet(x)
        x = self.end_layer(x)
        return x
