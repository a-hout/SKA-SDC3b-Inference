"""
Reduce dimensionality of PS2D while trying to retain the physics information -> small CNN
"""

import torch.nn as nn


class PS2DCompressor(nn.Module):
    """
    we want to turn the 10x10 matrix into a compressed and flattened vector array
    """

    def __init__(self, summary_dim: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=24, kernel_size=3
            ),  # TODO see if in channel 1 or 3, depending if each specific index has the three redshift ranges
            nn.GELU(),  # because lots of small values around 0 and thus we need a continuous function, this also allows us to preclude the need for log normalization
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=48 * 6 * 6, out_features=192),
            nn.GELU(),
            nn.Linear(
                192, summary_dim
            ),  # should experiment with summary_dim as a hyperparameter, for example [4, 8, 16, 32]
        )

    def forward(self, x):
        x = self.cnn(x)  # x of shape (batch, 3, 10, 10) if all is well
        x = self.fc(x)  # output will be the summary of shape (batch, summary_dim)
        return x
