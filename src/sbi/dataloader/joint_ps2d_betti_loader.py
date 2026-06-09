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
        valid_pairs,
        transform=None,
        target_transform=None,
    ):
        self.ps2d_dataset = ps2d_dataset
        self.betti_dataset = betti_dataset
        self.valid_pairs = valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        ps2d_index, betti_index = self.valid_pairs[idx]

        ps2d_theta, ps2d_obs = self.ps2d_dataset[ps2d_index]
        _, betti_obs = self.betti_dataset[betti_index]
        obs = torch.concatenate((torch.flatten(ps2d_obs), torch.flatten(betti_obs)))
        # obs is of shape (489) because (3, 10, 10) -> 300 and betti (3, 3, 21) -> 189
        return (ps2d_theta, obs)
