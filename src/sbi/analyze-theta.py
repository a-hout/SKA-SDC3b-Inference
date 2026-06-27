import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataloader.data_type_loader import DataDataset

OUTPUT_DIR = "plots/dataset_exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
print(f"using device: {device}\n")

datasets_to_explore = {
    "ps2d": {
        "path": "datasets/pk_128_train_250325_noise/memmap/",
        "loader_cls": lambda: (
            DataDataset
        ),  # wrapped in lambda to prevent import errors during setup
        # },
        # "betti": {
        #    "path": "datasets/betti/",
        #    "loader_cls": lambda: DataDataset,
    },
}

for name, config in datasets_to_explore.items():
    print(f"Analysing dataset {name.upper()}")
    try:
        dataset_cls = config["loader_cls"]()
        dataset = dataset_cls(name, device=device)
        _, obs = dataset[:]
        theta = torch.tensor(dataset.params[:, 0, :], device=device)

    except Exception as e:
        print(f"Skipping {name} due to loader/path error: {e}\n")
        continue

    theta = theta.cpu()
    theta_mean = theta.mean(dim=1)

    num_samples, num_bins = theta.shape
    print(f"Dataset shape: {num_samples} samples, {num_bins} redshift bins.")

    # Print Text Diagnostics to Terminal
    print("\nSummary statistics (mean parameter across bins)")
    print(f"Absolute min: {theta_mean.min().item():.5f}")
    print(f"16th percentile: {torch.quantile(theta_mean, 0.16).item():.5f}")
    print(f"50th percentile (median): {torch.median(theta_mean).item():.5f}")
    print(f"84th percentile: {torch.quantile(theta_mean, 0.84).item():.5f}")
    print(f"Absolute max: {theta_mean.max().item():.5f}")

    print("\nper bin statistics.........")
    for b in range(num_bins):
        bin_vals = theta[:, b]
        print(
            f"Bin {b + 1}\n\tmin: {bin_vals.min().item():.5f}\tmax: {bin_vals.max().item():.5f}\tmean: {bin_vals.mean().item():.5f}"
        )

        # check theta near boundar
        near_zero = (bin_vals <= 0.01).sum().item()
        near_one = (bin_vals >= 0.99).sum().item()
        if near_zero > 0 or near_one > 0:
            print(
                f"\tBoundary alert: {near_zero} samples <= 0.01, {near_one} samples >= 0.99"
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Theta analysis {name.upper()}", fontsize=16, fontweight="bold")

    # Panel 1 is th log-Scaled histogram of the mean
    axes[0, 0].hist(
        theta_mean.numpy(),
        bins=100,
        log=True,
        color="purple",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0, 0].axvline(
        theta_mean.min().item(),
        color="red",
        linestyle=":",
        label=f"Min ({theta_mean.min().item():.3f})",
    )
    axes[0, 0].axvline(
        theta_mean.max().item(),
        color="red",
        linestyle=":",
        label=f"Max ({theta_mean.max().item():.3f})",
    )
    axes[0, 0].set_title("Log-histogram of theta_mean")
    axes[0, 0].set_xlabel("theta_mean value")
    axes[0, 0].set_ylabel("Count (log scale)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2 is sorted index plot of the mean
    sorted_means, _ = torch.sort(theta_mean)
    axes[0, 1].plot(
        sorted_means.numpy(),
        marker=".",
        linestyle="none",
        markersize=2,
        color="darkblue",
        alpha=0.5,
    )
    axes[0, 1].axhline(
        torch.quantile(theta_mean, 0.16).item(),
        color="orange",
        linestyle="--",
        label="16th %",
    )
    axes[0, 1].axhline(
        torch.median(theta_mean).item(),
        color="green",
        linestyle="--",
        label="50th % (median)",
    )
    axes[0, 1].axhline(
        torch.quantile(theta_mean, 0.84).item(),
        color="blue",
        linestyle="--",
        label="84th %",
    )
    axes[0, 1].set_title("Sorted theta_mean")
    axes[0, 1].set_xlabel("Sorted data point index")
    axes[0, 1].set_ylabel("theta_mean value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # boxplot of
    # this will show if the distribution boundaries are identical across bins
    axes[1, 0].boxplot(
        [theta[:, b].numpy() for b in range(num_bins)],
        tick_labels=[f"Bin {b + 1}" for b in range(num_bins)],
    )
    axes[1, 0].set_title("Distribution spread per redshift bin")
    axes[1, 0].set_ylabel("xHI value")
    axes[1, 0].grid(True, alpha=0.3)

    # panel 4 is per-bin line overlayss for outliers
    sorted_idx = torch.argsort(theta_mean)
    lowest_indices = sorted_idx[:10]
    highest_indices = sorted_idx[-10:]

    for idx in lowest_indices:
        axes[1, 1].plot(
            range(1, num_bins + 1),
            theta[idx].numpy(),
            color="red",
            alpha=0.4,
            marker="o",
        )
    for idx in highest_indices:
        axes[1, 1].plot(
            range(1, num_bins + 1),
            theta[idx].numpy(),
            color="blue",
            alpha=0.4,
            marker="o",
        )

    axes[1, 1].plot([], [], color="red", label="Top 10 lowest mean histories")
    axes[1, 1].plot([], [], color="blue", label="Top 10 highest mean histories")
    axes[1, 1].set_title("Evolution history of extreme outliers")
    axes[1, 1].set_xlabel("Redshift bin")
    axes[1, 1].set_ylabel("xHI value")
    axes[1, 1].set_xticks(range(1, num_bins + 1))
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig_path = os.path.join(OUTPUT_DIR, f"{name}_diagnostic_report.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
