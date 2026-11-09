"""
script to load a posterior and run inference on it with new observations
"""
import benchmark_helper
import torch
import sys
import numpy as np
from sbi.inference import DirectPosterior


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
_ = torch.manual_seed(831) #random number for random generator

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
    OBS_TYPE = sys.argv[3].lower()   #
except IndexError:
    print(
        "Normal usage is: python src/sbi/benchmark_sbi_inference.py <model_path> <observation_path> <observation_type>"
    )
    sys.exit(1)

match OBS_TYPE:
    case "ps2d":
        print("Infering on PS2D observations...")

    case "betti":
        print("Infering on Betti numbers observations...")

    case "ps2d+betti":
        print("Inferring on both PS2D and Betti numbers...")

    case other:
        print("No such data type or data type combination has been trained.")
        sys.exit(1)

posterior = torch.load(
    MODEL_PATH,
    weights_only=False,
    map_location="cpu",
)
x_test = np.reshape(np.loadtxt("datasets/mock_ps2d_obs.txt"), (-1, 3, 10, 10))
print(x_test.shape)
#shape = (3, 10, 10) if OBS_TYPE=="ps2d" else (3, 3, 21) if OBS_TYPE=="betti" else (3, 10, 10)
#x_test = np.random.rand(10, shape[0], shape[1], shape[2]) #load observations in correct format ...? how
#x_test = torch.Tensor(x_test).to(device) maybe only with cuda

theta_pred, benchmark_theta_pred_sampling = benchmark_helper.theta_pred_sampling_benchmark(x_test, posterior)
theta_pred = theta_pred.numpy()
mean = theta_pred.mean(axis=0)
max = theta_pred.max(axis=0)
min = theta_pred.min(axis=0)

print(f"\n\nInference results for a window size of {x_test.shape[0]} {OBS_TYPE} observations")
print(f"{'-'*120}")
for max_i, mean_i, min_i, bin in zip(max, mean, min, ["low", "mid", "high"]):
    print(f"For {bin} reionization, \tin terms of xHI presence, there is a minimum of {min_i*100:.2f}%, a mean of {mean_i*100:.2f}%, and a maximum of {max_i*100:.2f}%")
print(f"{'-'*120}")
print(f"Time to infer: {benchmark_theta_pred_sampling[0]["total time sampling"]}")

"""
for _ in range(3):
    for _ in range(3):
        for _ in range(10):
            for _ in range(10):
                print(f"{[float(rando) for rando in np.random.rand(10)]}")
"""
