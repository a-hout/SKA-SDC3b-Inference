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
        self.transform = transform  # log10
        self.target_transform = target_transform

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        param = self.params[idx]
        obs = self.observables[idx]

        param_tensor = torch.tensor(
            param.copy(), device=self.device
        )  # copy because we want the arrays to be writable by the Tensor object

        obs_tensor = torch.tensor(obs.copy(), device=self.device)

        if self.transform:
            obs_tensor = self.transform(obs_tensor)
        if self.target_transform:
            param_tensor = self.target_transform(param_tensor)

        return (
            param_tensor,
            obs_tensor,
        )  # this will return a tuple of the parameter theta and the observable x at the specified index


if __name__ == "__main__":
    dataset = PS2DDataset("datasets/pk_128_train_250325_noise/memmap/")
    theta, x = dataset[:]
    print(x.min())  # min at -0.0395
