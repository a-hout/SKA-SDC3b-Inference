"""
This script will analyze both PS2D and Betti numbers

To swhocase the gaussianity or non gaussianity of both data types.

Left will show ps2d's image matrix to showcase it's variance in brightness temperature
Right will show the Betti number components counted across density thresholds

Chosen index 8000 as it represents about the middle of the reioniuzation
"""

import matplotlib.pyplot as plt
import numpy as np

#i like how we can do latex in labels and such for plots, should have seen that before

ps2d = np.load("datasets/ps2d/sbi2_ps2d_1000h.npy").astype(np.float32)
betti = np.load("datasets/betti/sbi2_betti_1000h.npy").astype(np.float32)

# pick an average universe index and middle redshift bin
idx = 8000 #can be something else doesnt matter
z_bin = 1
thresholds = np.linspace(-3, 3, betti.shape[-1])

#setup side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Comparing data types: frequency vs. topology", fontsize=16)

# left is ps2d
ax_ps = axes[0]
# vverage over the 20 noise realizations for a cleaner plot
ps2d_map = ps2d[idx].mean(axis=0)[z_bin]

im = ax_ps.imshow(
    np.log10(ps2d_map + 1e-10), #see the dataloader exampel
    cmap="viridis",
    origin="lower",
    aspect="auto",
)
ax_ps.set_title("PS2D", fontsize=12)
ax_ps.set_xlabel("$k_\\parallel$")
ax_ps.set_ylabel("$k_\\perp$")
fig.colorbar(im, ax=ax_ps, label="$\log_{10}$ power")

# add ps2d text
ax_ps.text(
    0.05, 0.85,
    "Measures scale/energy.\nNo info on physical shapes",
    transform=ax_ps.transAxes,
    color="white", fontsize=10,
    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
)

#betti
ax_bt = axes[1]
b0 = betti[idx].mean(axis=0)[z_bin, 0, :]
b1 = betti[idx].mean(axis=0)[z_bin, 1, :]
b2 = betti[idx].mean(axis=0)[z_bin, 2, :]

ax_bt.plot(thresholds, b0, label="$b_0$: components", color="darkorange", lw=2)
ax_bt.plot(thresholds, b1, label="$b_1$: holes", color="purple", lw=2)
ax_bt.plot(thresholds, b2, label="$b_2$: voids", color="teal", lw=2)

ax_bt.set_title("Betti", fontsize=12)
ax_bt.set_xlabel("Density threshold")
ax_bt.set_ylabel("Count")
ax_bt.legend(loc="upper right")
ax_bt.grid(True, alpha=0.3)

# add tekst to Betti
ax_bt.text(
    0.05, 0.45,
    "Captures non-Gaussian\nstructure of the IGM",
    transform=ax_bt.transAxes,
    color="black", fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

plt.tight_layout()
plt.savefig("plots/dataset_exploration/data_comparison.png", dpi=300)
plt.show()
