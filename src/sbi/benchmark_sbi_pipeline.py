"""
similar to the sbi_pipeline.py file but specific to benchmarking
"""

##################################################################################################################
# 0. imports
##################
import json
import sys
import mlflow
import benchmark_helper
import torch
from compressor import (
    betti_compressor,
    joint_ps2d_betti_compressor,
)
from dataloader import (
    data_type_loader,
    joint_ps2d_betti_loader,
)
from sbi.utils import BoxUniform
from torch.utils.data.dataset import Subset, random_split
import time

##################
# 1. inits
##################

# time benchmarking inits
benchmarks = {}
benchmark_total_pipeline_time_start = time.time()

pipeline_type = "singular"
full_realization = False #true if all 20 realization, else we should be picking a random realization for every observation

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
_ = torch.manual_seed(831) #random number for random generator

try:
    data_type = sys.argv[1].lower().strip()
except IndexError:
    print(
        "Normal usage is: python src/sbi/benchmark_sbi_pipeline.py <data_type> <optional best params path>"
    )
    exit(1)

try:
    with open(sys.argv[2]) as f:  # load path with best model parameters
        BEST_PARAMS = json.load(f)["best_params"]
except (IndexError, FileNotFoundError):
    print(
        "Normal usage is: python src/sbi/benchmark_sbi_pipeline.py <data_type> <best params path>"
    )
    exit(1)

dataset = None

match data_type:
    case "ps2d":
        mlflow.set_experiment("SBI PS2D")

    case "betti":
        mlflow.set_experiment("SBI Betti")

    case "ps2d+betti" | "betti+ps2d":  # combine both ps2d and betti
        data_type = "ps2d+betti"
        mlflow.set_experiment("SBI PS2D & Betti")

        dataset = joint_ps2d_betti_loader.JointPS2DBettiDataset(
            data_type_loader.DataDataset("ps2d", device=device),
            data_type_loader.DataDataset("betti", device=device),
        )

    case other:
        print("No such data type or data type combination.")
        exit(1)

if dataset is None:
    dataset = data_type_loader.DataDataset(data_type, device=device)

theta = torch.tensor(dataset.params[:, 0, :], device=device)
low = theta.min(dim=0).values.detach().clone()
high = theta.max(dim=0).values.detach().clone()
prior = BoxUniform(low=low, high=high, device=device) #have to watch out since NPE can have a bug where density prob is too close to prior bounds -> rejection smapling

benchmarks["data"] = {
    "data type": data_type,
    "load type": "full realization" if full_realization else "random realization",
    "pipeline": pipeline_type
}


def get_compressor(data_type, summary_dim, model=None):
    """
    fetches the embedding network to train with the normalizing flow (dependent on the data type)
    """
    match data_type:
        case "ps2d":
            return model
        case "betti":
            return betti_compressor.BettiCompressor(summary_dim=summary_dim)
        case "ps2d+betti":
            return joint_ps2d_betti_compressor.JointPS2DBettiCompressor(
                model,
                betti_compressor.BettiCompressor(summary_dim=summary_dim),
                summary_dim=summary_dim,
            )


global_gen = torch.manual_seed(831)

train_val_dataset, global_test_dataset = random_split(
    (Subset(dataset, range(int(len(dataset)*0.02)))), #when testing during demo since it's more rapid
    [0.8, 0.2],
    generator=global_gen,
)

##################
# 3. 30-run
##################
model_training_benchmark, model_info, (best_num, best_val_loss) = benchmark_helper.model_training_benchmark(BEST_PARAMS, mlflow, train_val_dataset, full_realization, get_compressor, data_type, device, prior, pipeline_type)
print(model_training_benchmark)

##################
# 4. best model
##################

best_density_estimator = dict(model_info["density_estimators"])[best_num]
best_posterior = dict(model_info["posteriors"])[best_num]
torch.save(
    best_density_estimator.state_dict(),
    f"models/sbi/{data_type}/best_density_estimator.pt",
)
torch.save(best_posterior, f"models/sbi/{data_type}/best_posterior.pt")

##################
# 5. eval
##################
model_testing_benchmark = benchmark_helper.model_testing_benchmark(mlflow, global_test_dataset, full_realization, best_posterior, best_val_loss, data_type, pipeline_type)
#save benchmark results
benchmarks["benchmarks"] = {
    "model training": model_training_benchmark,
    "model testing": model_testing_benchmark,
    "total pipeline time":float(time.time() - benchmark_total_pipeline_time_start)
}
benchmark_helper.save_benchmark(pipeline_type, data_type, benchmarks)
