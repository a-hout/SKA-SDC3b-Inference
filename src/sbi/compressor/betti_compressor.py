"""
Reduce dimensionality of Betti while trying to retain the physics information -> MLP
"""

import torch.nn as nn


class BettiCompressor(nn.Module):
    """
    we want to turn the 3-point statistic into a compressed and flattened vector array
    """

    def __init__(self, summary_dim: int = 4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 21, out_features=128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(
                0.1
            ),  # avoid overfitting on the noise (may be more useful for low observation time)
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(
                64, summary_dim
            ),  # should experiment with summary_dim as a hyperparameter, for example [4, 8, 16, 32]
        )

    def forward(self, x):
        # shape (batch, 20, 3, 3, 21), quite complex so need to reduce
        x = x.mean(
            dim=1
        )  # we average the noise realisations (same logic already applied for ps2d) to reduce dimensionality
        # shape (batch, 3, 3, 21)
        x = x.flatten(start_dim=1)  # shape (batch, 189)
        x = self.fc(
            x
        )  # x of shape (batch, 189) if all is well, and output will be the summary of shape (batch, summary_dim)
        return x
