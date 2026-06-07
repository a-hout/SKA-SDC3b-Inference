"""
multi-modal compressor, supposed to work with other compressors, get their summary statistics, run a small FC and return its own vector
"""

import torch.nn as nn


class CombineCompressor(nn.Module):
    def __init__(self, embeddings, summary_dim: int = 4):
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
        # shape (batch, 489)

        # one part (first 300) of x is ps2d, which we will feed to the ps2d embedding

        # other part (next 189) of x is betti, which we will feed to the betti embedding

        # combine the x and feed to fully connected network
        x = self.fc(
            x
        )  # x of shape (batch, 189) if all is well, and output will be the summary of shape (batch, summary_dim)
        return x
