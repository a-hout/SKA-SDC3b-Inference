"""
transform the hundreds of thousands of inputs and targets of pk_128_train into a numpy memmap for ease of loading during training
"""

import numpy as np

# file name conventions:
base_input = "datasets/pk_128_train_250325_noise/inputs/PS0_i"
hz_ranges = ["151.0_165.9", "166.0_180.9", "181.0_195.9"]

base_target = "datasets/pk_128_train_250325_noise/targets/ytrue_pk_i"

x_memmap = np.memmap(
    "datasets/pk_128_train_250325_noise/memmap/x_memmap.npy",
    dtype=np.float32,
    mode="w+",
    shape=(159449, 3, 10, 10),
)
theta_memmap = np.memmap(
    "datasets/pk_128_train_250325_noise/memmap/theta_memmap.npy",
    dtype=np.float32,
    mode="w+",
    shape=(159449, 3),
)

sum_files_not_loaded = 0
for id in range(159449 + 1):
    try:
        # for theta
        theta_i = np.loadtxt(base_target + str(id) + ".txt")
        theta_memmap[id] = theta_i[:, -1]  # mean_xHI as theta
        x_i = []
        for hz in hz_ranges:
            x_i.append(np.loadtxt(base_input + "0" + "_" + hz + ".txt"))
        x_memmap[id] = np.array(x_i)
    except:
        sum_files_not_loaded += 1

del x_memmap
del theta_memmap

print("files that could not be loaded: ", sum_files_not_loaded)
