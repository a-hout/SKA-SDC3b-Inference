"""
Reduce dimensionality of PS2D while trying to retain the physics information -> small CNN
"""

import torch.nn as nn


class PS2DCompressor(nn.Module):
    """
    we want to turn the 10x10 matrix into a compressed and flattened vector array
    """

    def __init__(self, summary_dim: int = 4, act_fun: type = nn.GELU):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, padding=1
            ),  # padding because 10x10 is quite small and we want to preserve the dimension as much as possible while extracting characteristics
            nn.BatchNorm2d(32),
            act_fun(),
            # nn.GELU(),  # because lots of small values around 0 and thus we need a continuous function, this also allows us to preclude the need for log normalization
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            act_fun(),
            # nn.GELU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 8, out_features=256),
            nn.LayerNorm(256),
            act_fun(),
            # nn.GELU(),
            nn.Dropout(
                0.1
            ),  # avoid overfitting on the noise (may be more useful for low observation time)
            nn.Linear(256, 64),
            act_fun(),
            # nn.GELU(),
            nn.Linear(
                64, summary_dim
            ),  # should experiment with summary_dim as a hyperparameter, for example [4, 8, 16, 32]
        )

    def forward(self, x):
        x = self.cnn(x)  # x of shape (batch, 3, 10, 10) if all is well
        x = self.fc(x)  # output will be the summary of shape (batch, summary_dim)
        return x
