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
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.GELU(),  # because lots of small values around 0 and thus we need a continuous function, this also allows us to preclude the need for log normalization
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 6 * 6, out_features=64),
            nn.GELU(),
            nn.Linear(
                64, summary_dim
            ),  # should experiment with summary_dim as a hyperparameter
        )

    def forward(self, x):
        x = self.cnn(x)  # need to make sure x is reshaped correctly
        x = self.fc(x)  # output will be the summary
        return x
