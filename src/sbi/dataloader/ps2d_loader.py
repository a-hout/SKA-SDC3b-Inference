import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class PS2DDataset(Dataset):
    def __init__(
        self, memmap_path, transform=None, target_transform=None, device="cpu"
    ):
        self.device = device
        self.params = np.memmap(
            memmap_path + "theta_memmap.npy",
            shape=(159449, 3),
            mode="r",
            dtype=np.float32,
        )
        self.observables = np.memmap(
            memmap_path + "x_memmap.npy",
            shape=(159449, 3, 10, 10),
            mode="r",
            dtype=np.float32,
        )
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
    dataset = PS2DDataset("datasets/pk_128_train_250325_noise/memmap/")
    theta, x = dataset[0]
    print(theta.shape, x.shape)  # here expect (3,) and (3, 10, 10)
