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


def infer_parameters(OBS_TYPE, OBS_PATH, MODEL_PATH):
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
        map_location="cpu",
    )
    if OBS_TYPE != "ps2d+betti":
        shape = (3, 10, 10) if OBS_TYPE=="ps2d" else (3, 3, 21) # if OBS_TYPE=="betti" else (3, 10, 10)
        x_test = np.reshape(np.loadtxt(OBS_PATH), (-1, shape[0], shape[1], shape[2]))
    else:
        ps2d_obs = torch.as_tensor(np.reshape(np.loadtxt(OBS_PATH[0]), (-1, 3, 10, 10)), dtype=torch.float32, device=device)
        betti_obs = torch.as_tensor(np.reshape(np.loadtxt(OBS_PATH[1]), (-1, 3, 3, 21)), dtype=torch.float32, device=device)
        ps2d_flat = ps2d_obs.view(ps2d_obs.shape[0], -1)
        betti_flat = betti_obs.view(betti_obs.shape[0], -1)

        x_test = torch.cat((ps2d_flat, betti_flat), dim=1)
    x_test = torch.as_tensor(x_test, dtype=torch.float32, device=device)

    theta_pred = []
    bench_batch = {}
    batch_size_list = [1, 16, 32, 64, 128, 256, int(x_test.shape[0]*0.01), int(x_test.shape[0]*0.05), int(x_test.shape[0]*0.10), int(x_test.shape[0]*0.25), int(x_test.shape[0]*0.5)]
    for batch_size in batch_size_list:
        theta_pred = []
        bench_start = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(x_test), batch_size), desc="Sampling test observations"):
                x_batch = x_test[i:i + batch_size]
                samples = posterior.sample_batched(
                    (1000,),
                    x=x_batch,
                    show_progress_bars=False,
                )
                theta_pred.append(samples.mean(dim=0).cpu())

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        theta_pred = torch.cat(theta_pred)
        bench_batch[batch_size] = time.time() - bench_start

    #theta_pred, benchmark_theta_pred_sampling = benchmark_helper.theta_pred_sampling_benchmark(x_test, posterior)

    theta_pred = theta_pred.numpy()
    mean = theta_pred.mean(axis=0)
    max = theta_pred.max(axis=0)
    min = theta_pred.min(axis=0)

    print(f"\n\nInference results for a window size of {x_test.shape[0]} {OBS_TYPE} observations")
    print(f"{'-'*120}")
    for max_i, mean_i, min_i, bin in zip(max, mean, min, ["low", "mid", "high"]):
        print(f"For {bin} reionization, \tin terms of xHI presence, there is a minimum of {min_i*100:.2f}%, a mean of {mean_i*100:.2f}%, and a maximum of {max_i*100:.2f}%")
    print(f"{'-'*120}")
    print(f"Time to infer batched sampling: {bench_batch}")
    #print(f"Time to infer sampling per obs: {benchmark_theta_pred_sampling[0]["total time sampling"]}")
    return bench_batch

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        #else "mps"
        #if torch.mps.is_available()
        else "cpu"
    )
    _ = torch.manual_seed(831) #random number for random generator

    """
    try:
        MODEL_PATH = sys.argv[1]
    except IndexError:
        print(
            "Normal usage is: python src/sbi/benchmark_sbi_inference.py <model_path> <observation_path> <observation_type>"
        )
        sys.exit(1)

    try:
        OBS_PATH = sys.argv[2]   # load path with observations
    except IndexError:
        print(
            "Normal usage is: python src/sbi/benchmark_sbi_inference.py <model_path> <observation_path> <observation_type>"
        )
        sys.exit(1)

    try:
        OBS_TYPE = sys.argv[3].lower()
    except IndexError:
        print(
            "Normal usage is: python src/sbi/benchmark_sbi_inference.py <model_path> <observation_path> <observation_type>"
        )
        sys.exit(1)


    ambiguous_path = MODEL_PATH.split('/')
    model = ambiguous_path[1]
    ambiguous_path.remove(MODEL_PATH.split('/')[1])
    other_model = "sbi" if "ensemble-sbi"==model else "ensemble-sbi"
    ambiguous_path.insert(1, other_model)
    other_path = '/'.join(ambiguous_path)
    paths = [MODEL_PATH, other_model]

    """
    OBS_PATH = ["datasets/mock_ps2d_obs_1000.txt", "datasets/mock_betti_obs_1000.txt"]
    obs_len = 1000
    OBS_TYPE = ["ps2d", "betti", "ps2d+betti"]

    for _ in range(5):
        benchmarks = {"ps2d": {},"betti": {},"ps2d+betti": {},}

        for data_type in OBS_TYPE:
            benchmarks[data_type]["singular"] = infer_parameters(data_type, OBS_PATH[0] if data_type=="ps2d" else OBS_PATH[1] if data_type=="betti" else OBS_PATH, f"models/sbi/{data_type}/best_posterior.pt")
            benchmarks[data_type]["ensemble"] = infer_parameters(data_type, OBS_PATH[0] if data_type=="ps2d" else OBS_PATH[1] if data_type=="betti" else OBS_PATH, f"models/ensemble-sbi/{data_type}/best_posterior.pt")


        local_time = time.localtime() #to get day-hour-min
        if not os.path.isdir("benchmarks"):
            os.mkdir("benchmarks")

        with open(f"benchmarks/{device}-inferring-{obs_len}-obs-benchmark-{local_time.tm_mday}-{local_time.tm_hour}-{local_time.tm_min}.json", "w") as f: #timestamp at the end in case we want multiple benchmarks of the same pipeline/data
            json.dump(
                benchmarks,
                f,
                indent=4,
            )

        pprint(benchmarks)
    """
    betti random mock observations
    for _ in range(100):
        for _ in range(3):
            for _ in range(3):
                    print(f"{[float(rando) for rando in np.random.rand(21)]}")

    ps2d random mock observations
    for _ in range(100):
        for _ in range(3):
            for _ in range(10):
                    print(f"{[float(rando) for rando in np.random.rand(10)]}")
    """
