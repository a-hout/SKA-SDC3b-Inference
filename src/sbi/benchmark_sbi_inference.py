"""
benchmark script to load a posterior and run inference on it with new (mock) observations:
1. inference time for 1000 observations (with different batch sizes)
2. all data types and posteriors (6)
3. CPU vs CUDA
"""

from pprint import pprint

import json
import os
import benchmark_helper
import torch
import sys
import time
import numpy as np
from sbi.inference import DirectPosterior
from tqdm import tqdm


def infer_parameters(OBS_TYPE, OBS_PATH, MODEL_PATH, device):
    match OBS_TYPE:
        case "ps2d":
            print("\nInfering on PS2D observations...\n")

        case "betti":
            print("\nInfering on Betti numbers observations...\n")

        case "ps2d+betti":
            print("\nInferring on both PS2D and Betti numbers...\n")

        case other:
            print("No such data type or data type combination has been trained.")
            sys.exit(1)

    posterior = torch.load(
        MODEL_PATH,
        weights_only=False,
        map_location=device,
    )
    if OBS_TYPE != "ps2d+betti":
        shape = (3, 10, 10) if OBS_TYPE=="ps2d" else (3, 3, 21)
        x_test = np.reshape(np.loadtxt(OBS_PATH), (-1, shape[0], shape[1], shape[2]))
    else:
        ps2d_obs = torch.as_tensor(np.reshape(np.loadtxt(OBS_PATH[0]), (-1, 3, 10, 10)), dtype=torch.float32, device=device)
        betti_obs = torch.as_tensor(np.reshape(np.loadtxt(OBS_PATH[1]), (-1, 3, 3, 21)), dtype=torch.float32, device=device)
        ps2d_flat = ps2d_obs.view(ps2d_obs.shape[0], -1)
        betti_flat = betti_obs.view(betti_obs.shape[0], -1)

        x_test = torch.cat((ps2d_flat, betti_flat), dim=1)
    x_test = torch.as_tensor(x_test, dtype=torch.float32, device=device)

    start = time.time()
    with torch.no_grad():
        samples = posterior.sample((10000,), x=x_test[0])
        theta_pred = samples.mean(dim=0).to(device).numpy()

    return time.time() - start

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    _ = torch.manual_seed(831) #random number for random generator

    OBS_PATH = ["datasets/mock_ps2d_obs_1000.txt", "datasets/mock_betti_obs_1000.txt"]
    OBS_TYPE = ["ps2d", "betti", "ps2d+betti"]

    for i in range(30):
        benchmarks = {"ps2d": {},"betti": {},"ps2d+betti": {},}

        for data_type in OBS_TYPE:
            benchmarks[data_type]["singular"] = infer_parameters(data_type, OBS_PATH[0] if data_type=="ps2d" else OBS_PATH[1] if data_type=="betti" else OBS_PATH, f"models/sbi/{data_type}/best_posterior.pt", device)
            benchmarks[data_type]["ensemble"] = infer_parameters(data_type, OBS_PATH[0] if data_type=="ps2d" else OBS_PATH[1] if data_type=="betti" else OBS_PATH, f"models/ensemble-sbi/{data_type}/best_posterior.pt", device)


        local_time = time.localtime() #to get day-hour-min
        if not os.path.isdir("benchmarks"):
            os.mkdir("benchmarks")
        if not os.path.isdir("benchmarks/inferring"):
            os.mkdir("benchmarks/inferring")

        if not os.path.isdir(f"benchmarks/inferring/{device}"):
            os.mkdir(f"benchmarks/inferring/{device}")

        with open(f"benchmarks/inferring/{device}/{i}.json", "w") as f:
            json.dump(
                benchmarks,
                f,
                indent=4,
            )

        pprint(benchmarks)
