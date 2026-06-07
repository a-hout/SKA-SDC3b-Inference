import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data.dataset import Dataset


class BettiDataset(Dataset):
    def __init__(
        self, memmap_path, transform=None, target_transform=None, device="cpu"
    ):
        self.device = device
        self.params = np.load(
            memmap_path + "sbi2_y.npy",
        ).astype(np.float32)
        raw_observables = np.load(
            memmap_path + "sbi2_betti_1000h.npy",
        ).astype(np.float32)

        # inbuilt robust scaler (testing for now)
        if transform:
            original_shape = raw_observables.shape
            num_samples = original_shape[0]

            flattened_obs = raw_observables.reshape(
                num_samples, -1
            )  # 2d for scikit-learn

            self.scaler = RobustScaler()
            scaled_flattened = self.scaler.fit_transform(flattened_obs)

            self.observables = scaled_flattened.reshape(
                original_shape
            )  # back to original shape
        else:
            self.observables = raw_observables
            self.scaler = None

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        param = self.params[idx]
        obs = self.observables[idx]
        random_noise = np.random.randint(
            0, obs.shape[0]
        )  # get index of random noise realisation (between 0 and 21)

        return (
            torch.tensor(
                param[0].copy(), device=self.device
            ),  # first element since the 20 noise realisations all share the same theta -> shape (3, )
            torch.tensor(
                obs[random_noise].copy(), device=self.device
            ),  # -> shape (3, 3, 21)
        )  # this will return a tuple of the parameter theta and the observable x at the specified index


if __name__ == "__main__":
    dataset = BettiDataset("datasets/betti/")
    theta, obs = dataset[:]
    _, obs = dataset[0]
    print("theta and obs shape:", theta.shape, obs.shape)  # here expect
    N, N_noise, N_freqs = obs.shape[0], obs.shape[1], obs.shape[2]
    dim_data = obs.shape[3:]
    dim_out = theta.shape[0]
    print("N", N)
    print("N_noise", N_noise)
    print("N_freqs", N_freqs)
    print("dim_data", dim_data)
    print("dim_out", dim_out)

    true_thetas = torch.tensor(dataset.params[:, 0, :], device=dataset.device)
    print("True global theta shape:", true_thetas.shape)  # Should be (15951, 3)

    print("True global theta shape:", true_thetas.shape)  # Should be (15951, 3)

    theta_low = true_thetas.min(dim=0).values.detach().clone()
    theta_high = true_thetas.max(dim=0).values.detach().clone()

    print("Global low: ", theta_low)
    print("Global high: ", theta_high)
