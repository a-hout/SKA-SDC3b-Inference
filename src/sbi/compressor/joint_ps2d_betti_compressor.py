"""
multi-modal compressor, supposed to work with other compressors, get their summary statistics, run a small FC and return its own vector
"""

import torch
import torch.nn as nn


class JointPS2DBettiCompressor(nn.Module):
    def __init__(
        self,
        ps2d_embedding: type,
        betti_embedding: type,
        summary_dim: int = 4,
    ):
        super().__init__()

        self.ps2d_embedding = ps2d_embedding
        self.betti_embedding = betti_embedding
        self.fc = nn.Sequential(
            nn.Linear(in_features=summary_dim + 3, out_features=128),
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
        # shape (batch, 489)

        # one part (first 300) of x is ps2d, which we will feed to the ps2d embedding
        x_ps2d = x[:, :300].view(
            -1, 3, 10, 10
        )  # fit to number of samples and basic shape
        x_ps2d = self.ps2d_embedding(x_ps2d)

        # other part (next 189) of x is betti, which we will feed to the betti embedding
        x_betti = x[:, 300:].view(-1, 3, 3, 21)
        x_betti = self.betti_embedding(x_betti)

        # combine the x and feed to fully connected network

        x = torch.concatenate((x_ps2d, x_betti), dim=1)

        x = self.fc(x)
        return x
