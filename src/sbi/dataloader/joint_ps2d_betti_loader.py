"""
dataloader for the generic datadataloader class (passed onto), needs to fetch the observations of both, concatenate them to create ONE observation tensor (20, 489)
"""

import torch
from torch.utils.data.dataset import Dataset


class JointPS2DBettiDataset(Dataset):
    """
    wrapper class so we can get ps2d and betti obs with the same theta
    """

    def __init__(
        self,
        ps2d_dataset,
        betti_dataset,
        transform=None,
        target_transform=None,
    ):
        self.ps2d_dataset = ps2d_dataset
        self.betti_dataset = betti_dataset
        self.params = ps2d_dataset.params

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        ps2d_theta, ps2d_obs = self.ps2d_dataset[idx]
        _, betti_obs = self.betti_dataset[idx] #dont need theta it's the sae as ps2d
        ps2d_flat = ps2d_obs.view(ps2d_obs.shape[0], -1)  #(20, 300) ps2d
        betti_flat = betti_obs.view(betti_obs.shape[0], -1)  #  (20, 189) betti

        obs = torch.cat((ps2d_flat, betti_flat), dim=1)  # (20, 489)
        # obs is of shape (20, 489) because (20, 3, 10, 10) -> 20*300 and betti (20, 3, 3, 21) -> 20*189
        return (ps2d_theta, obs)
