"""
utilities for benchmarking the pipelines
"""
import json
import os
import time
import signal
import pprint
import numpy as np
import torch
from torch.utils.data.dataset import Subset, random_split
from sbi.diagnostics import check_tarp, run_tarp

def data_loading_benchmark(gen, train_val_dataset, benchmark_data_loading, full_realization=True):
    """
    If full realization is set to false, then we randomly pick a realization TODO
    """
    benchmark_start = time.time()

    if full_realization:
        train_set, test_set = random_split(
            train_val_dataset, [0.8, 0.2], generator=gen
        )

        # get all items from the dataset in one pass this is C++ optimized in the backend
        # train_set[j] returns (param shape=(3,), obs shape=(20, 3, 3, 21) for betti)
        all_params, all_obs = zip(*[train_set[j] for j in range(len(train_set))])
        all_params = torch.stack(all_params)  # (N, 3)  ex -> (12000, 3)
        all_obs = torch.stack(all_obs)     # (N, 20, dot dot dot) ex -> (12000, 20, 3, 3, 21)

        #exxpand theta to match every noise realization
        N, n_noise = all_obs.shape[:2]  #N=12000, n_noise=20 (i think?)

        # unsquueeze adds a dimension so(12000, 3) -> (12000, 1, 3)
        # expand copies it 20 times so (12000, 1, 3) -> (12000, 20, 3)
        # reshpae flattens N*noise into one so (12000, 20, 3) -> (240000, 3) big array!
        theta_train = all_params.unsqueeze(1).expand(N, n_noise, -1).reshape(N * n_noise, -1)

        #dlatten obs the same way
        # (12000, 20, 3, 3, 21) -> (240000, 3, 3, 21) (for betti, same logic for ps2d)
        x_train = all_obs.reshape(N * n_noise, *all_obs.shape[2:])

        all_params, all_obs = zip(*[test_set[j] for j in range(len(test_set))])
        all_params = torch.stack(all_params)
        all_obs = torch.stack(all_obs)

        N, n_noise = all_obs.shape[:2]
        theta_test = all_params.unsqueeze(1).expand(N, n_noise, -1).reshape(N * n_noise, -1)
        x_test = all_obs.reshape(N * n_noise, *all_obs.shape[2:])

    benchmark_data_loading.append(time.time()-benchmark_start)

    return theta_train, x_train, theta_test, x_test, benchmark_data_loading

def data_loading_benchmark_measures(benchmark_data_loading):
    benchmark_data_loading = np.array(benchmark_data_loading)
    data_loading_benchmarks = {
        "average load time": float(benchmark_data_loading.mean()),
        "max load time": float(benchmark_data_loading.max()),
        "min load time": float(benchmark_data_loading.min()),
        "total time": float(benchmark_data_loading.sum()),
    },

    return data_loading_benchmarks

def tarp_benchmark(theta_test, x_test, posterior, benchmark_tarp_20_realizations: list):
    def timeout_handler(signum, frame):
        """
        we need to kill the TARPing if it's >10 minutes (bad luck due to bad posterior/prior sampling)
        """
        raise TimeoutError("TARP timed out")

    # 20 realisations
    mean_pval_tarp = 0
    successful_tarps = 0
    current_model_curves = []  # for the 20 tarps
    for _ in range(20):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes

        try:
            benchmark_start = time.time()

            num_tarp_samples = 500
            tarp_index = torch.randperm(len(theta_test))[:num_tarp_samples]

            prior_samples = theta_test[tarp_index]  # ground truth theta
            prior_predictives = x_test[tarp_index]  # corresponding observations

            ecp, alpha = run_tarp(
                prior_samples,
                prior_predictives,
                posterior,
                references=None,
                num_posterior_samples=1000,
                use_batched_sampling=True,
            )

            _, ks_pval = check_tarp(ecp.cpu(), alpha.cpu())
            signal.alarm(0)
            benchmark_tarp_20_realizations.append(time.time()-benchmark_start) #one tarp realisation

            mean_pval_tarp += ks_pval
            successful_tarps += 1

            current_model_curves.append(
                (ecp.cpu().numpy(), alpha.cpu().numpy())
            )

        except TimeoutError:
            print(
                "could not complete tarp realisation, skipping to next one..."
            )
            continue  # go to the next run

    if successful_tarps > 0:
        mean_pval_tarp /= successful_tarps
    else:
        mean_pval_tarp = 0.0  # very bad!

    return mean_pval_tarp, current_model_curves, benchmark_tarp_20_realizations

def tarp_benchmark_measures(benchmark_tarp_20_realizations: list):

    chunked_20_realizations = np.array(benchmark_tarp_20_realizations).reshape((int(len(benchmark_tarp_20_realizations)/20), 20)) #normally 20, 30 but this is more robust incase not 30 models

    tarp_benchmarks = {
        "fastest TARP realization": float(min(benchmark_tarp_20_realizations)),
        "slowest TARP realization": float(max(benchmark_tarp_20_realizations)),
        "20 TARP min time": float(chunked_20_realizations.sum(axis=1).min()),
        "20 TARP max time": float(chunked_20_realizations.sum(axis=1).max()),
        "20 TARP average time": float(chunked_20_realizations.sum(axis=1).mean()),
        "20 TARP average time for one realization": float(chunked_20_realizations.mean()), #get the average of 20 realizations of the average of one realization
        "20 TARP total time": float(sum(benchmark_tarp_20_realizations)), #the total time spent TARPing
    }

    return tarp_benchmarks

def save_benchmark(pipeline_type, data_type, benchmarks):
    """
    benchmarks should follow the structure found in the bottom of this script
    """

    local_time = time.localtime() #to get day-hour-min

    if not os.path.isdir("benchmarks"):
        os.mkdir("benchmarks")

    with open(f"benchmarks/{pipeline_type}-{data_type}-benchmark-{local_time.tm_mday}-{local_time.tm_hour}-{local_time.tm_min}.json", "w") as f: #timestamp at the end in case we want multiple benchmarks of the same pipeline/data
        json.dump(
            {
                benchmarks
            },
            f,
            indent=4,
        )

"""
TODO WIP, still need to discuss what is valuable to benchmark

benchmarks = {
    "data":{ #this block is to identify the data
        "data type": data_type, #ps2d, betti or ps2d+betti
        "load type", #full realization or random realization
        "shape of obs", #shape of given data type's observation
        "number of obs", #number of observations
        "shape of theta", #shape of the theta parameter
        "number of theta",
        "pipeline", #either singular or ensemble
    },
    "benchmarks":{
        "gridsearching":{
            "average time per parameter",
            "max time for a parameter",
            "min time for a parameter",
            "average cv time",
            "max cv time",
            "min cv time",
            "total gridsearching time",
        },
        "model training":{
            "training time": {
                "total run time", #including drawing figures and finding the best model
                "total training time", #not including the above
                "average training time",
                "max training time",
                "min training time",
            },
            "data loading":{
                "average load time",
                "max load time",
                "min load time",
                "total time",
            },
            "embedding":{
                "betti":{ #only for betti
                    "forward pass time"
                },
                "resnet":{ #only for ps2d and ps2d+betti
                    "average epoch time",
                    "total time",
                },
            }
            "density estimator training":{
                "average training time",
                "max training time",
                "min training time",
                "average training time per epoch",
            },
            "TARP":{
                "fastest TARP realization",
                "slowest TARP realization",
                "20 TARP min time",
                "20 TARP max time",
                "20 TARP average time",
                "20 TARP average time for one realization", #get the average of 20 realizations of the average of one realization
                "20 TARP total time", #the total time spent TARPing
            },
        "model testing": {
            "data preparation":{
                "load time",
            },
            "sampling theta_pred":{
                "average sample time",
                "max sample time",
                "min sample time",
                "total time",
            },
            "model metrics":{
                "per bin":{
                    "average load time",
                    "max load time",
                    "min load time",
                    "total time",
                },
                "overall metrics time",
            },
            "reionization history and pairplots":{
                "low time",
                "mid time",
                "high time",
                "average time",
                "min time",
                "max time",
                "total time",
            },
            "uncertainty":{
                "low time",
                "mid time",
                "high time",
                "average time",
                "min time",
                "max time",
                "total time",
            }
            "global TARP":{
                "20 TARP min time",
                "20 TARP max time",
                "20 TARP average time for one realization",
                "20 TARP total time",
                "TARP draw figure time",
            },
            "PPC":{
                "low PPC time",
                "mid PPC time",
                "high PPC time",
                "average PPC time",
                "min PPC time",
                "max PPC time",
                "total PPC time",
            },
            "total testing time",
        },
        "total pipeline time",
    }
}
"""


if __name__ == "__main__":
    fake_tarps = [1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,6,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,7,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,6,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,9,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,1,1,2,1,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,2,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,3,4,3,1,2,3,
        1,2,3,2,1,2,3,2,1,2,4,3,1,2,5,4,3,1,2,3,]
    #tarp_benchmark_measures(fake_tarps)

    fake_data_loading = [15, 13, 12, 11, 19, 11, 10, 6,12,6,8,3,12,5,2,1,11,4,1,6,4,8,5,8,9,23,2,11,12,13]
    #data_loading_benchmark_measures(fake_data_loading)
