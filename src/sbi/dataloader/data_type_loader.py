"""
This is a generic data loader for both betti and ps2d, which will fetch the respective observations (luckily they have about the same-ish shape)
and also fetch the corresponding theta parameter

If need be, the 1000h can be changed to 100h since the dataset has that integration observation time as well
"""

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DataDataset(Dataset):
    def __init__(self, type: str, transform=None, target_transform=None, device="cpu"):
        obs_path = f"datasets/{type}/sbi2_{type}_1000h.npy"
        self.device = device

        self.params = np.load(
            "datasets/sbi2_y.npy",
        ).astype(np.float32)
        self.observables = np.load(
            obs_path,
        ).astype(np.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        param = self.params[idx]
        obs = self.observables[idx]

        return (
            torch.tensor(
                param[0].copy(), device=self.device
            ),  # first element since the 20 noise realisations all share the same theta -> shape (3, )
            torch.tensor(
                obs.copy(), device=self.device
            ),  # -> shape (20, 3, 3, 21) if betti, else (20, 3, 10, 10) if ps2d
        )  # this will return a tuple of the parameter theta and the observable x at the specified index


if __name__ == "__main__":
    for data_type in ["ps2d", "betti"]:
        print(f" SURVEYING DATASET: {data_type.upper()}")

        #load dataset object
        dataset = DataDataset(data_type)

        #acess  numpy arrays directly
        obs = dataset.observables# (N, 20, 3, 10, 10) or (N, 20, 3, 3, 21)
        params = dataset.params[:, 0, :] #(N, 3)

        print(f"Observables shape:\t{obs.shape}")
        print(f"Params shape:\t{params.shape}")

        # global observational statistics
        print(f"Global min:\t{obs.min():12.4f}")
        print(f"Global max:\t{obs.max():12.4f}")
        print(f"Global mean:\t{obs.mean():12.4f}")
        print(f"Global std:\t{obs.std():12.4f}")
        print(f"Has NaNs / infs:\t{np.isnan(obs).any() or np.isinf(obs).any()}")

        # scenario breakdown (low, mid, high based on theta values)
        # we group by the mean parameter value across redshift bins
        mean_theta_per_sample = params.mean(axis=1)
        p16, p84 = np.percentile(mean_theta_per_sample, [16, 84])

        low_mask = mean_theta_per_sample <= p16
        mid_mask = (mean_theta_per_sample > p16) & (mean_theta_per_sample < p84)
        high_mask = mean_theta_per_sample >= p84

        print("Scenario observational statistics:")
        print(f"  [LOW  (fast)]  Min: {obs[low_mask].min():10.2f} | Max: {obs[low_mask].max():10.2f} | Mean: {obs[low_mask].mean():10.2f}")
        print(f"  [MID  (avg) ]  Min: {obs[mid_mask].min():10.2f} | Max: {obs[mid_mask].max():10.2f} | Mean: {obs[mid_mask].mean():10.2f}")
        print(f"  [HIGH (slow)]  Min: {obs[high_mask].min():10.2f} | Max: {obs[high_mask].max():10.2f} | Mean: {obs[high_mask].mean():10.2f}")
        print("\n")
