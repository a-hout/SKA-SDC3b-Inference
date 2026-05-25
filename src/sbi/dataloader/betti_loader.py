import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class BettiDataset(Dataset):
    def __init__(
        self, memmap_path, transform=None, target_transform=None, device="cpu"
    ):
        self.device = device
        self.params = np.load(
            memmap_path + "sbi2_y.npy",
        ).astype(np.float32)
        self.observables = np.load(
            memmap_path + "sbi2_betti_1000h.npy",
        ).astype(np.float32)
        self.transform = transform  # no transform for now.
        self.target_transform = target_transform

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        param = self.params[idx]
        obs = self.observables[idx]
        """
        if self.transform:
            obs = self.transform(obs)
        if self.target_transform:
            param = self.target_transform(param)
        """

        return (
            torch.tensor(
                param.copy(), device=self.device
            ),  # copy because we want the arrays to be writable by the Tensor object
            torch.tensor(obs.copy(), device=self.device),
        )  # this will return a tuple of the parameter theta and the observable x at the specified index


if __name__ == "__main__":
    dataset = BettiDataset("datasets/betti/")
    theta, obs = dataset[:]
    print("theta and obs shape:", theta.shape, obs.shape)  # here expect
    N, N_noise, N_freqs = obs.shape[0], obs.shape[1], obs.shape[2]
    dim_data = obs.shape[3:]
    dim_out = theta.shape[2]

    print("N", N)
    print("N_noise", N_noise)
    print("N_freqs", N_freqs)
    print("dim_data", dim_data)
    print("dim_out", dim_out)

    print("low: ", theta.min(dim=0).values.detach().clone())
    print("high: ", theta.min(dim=0).values.detach().clone())
