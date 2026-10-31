"""
utilities for benchmarking the pipelines
"""
import json
import os
import time
import signal
import pprint
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Subset, random_split
from sbi.diagnostics import check_tarp, run_tarp
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

#---
#model training
#---

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

#---
#model testing
#---

def test_data_loading_benchmark(global_test_dataset, full_realization=True):
    benchmark_test_data = time.time()
    if full_realization:
        all_params, all_obs = zip(*[global_test_dataset[j] for j in range(len(global_test_dataset))])
        all_params = torch.stack(all_params)
        all_obs = torch.stack(all_obs)

        N, n_noise = all_obs.shape[:2]
        theta_test = all_params.unsqueeze(1).expand(N, n_noise, -1).reshape(N * n_noise, -1)
        x_test = all_obs.reshape(N * n_noise, *all_obs.shape[2:])

    return all_params, x_test, theta_test, n_noise, time.time() - benchmark_test_data

def theta_pred_sampling_benchmark(x_test_metrics, best_posterior):
    benchmark_theta_pred_samples = []

    theta_pred = []
    for x_obs in tqdm(x_test_metrics, desc="Sampling test observations"):
        benchmark_start = time.time()
        with torch.no_grad():
            samples = best_posterior.sample(
                (1000,),
                x=x_obs,
                show_progress_bars=False,
            )  # 1000 samples from the approximate distribution given an observation

            theta_pred.append(
                samples.mean(dim=0).cpu()
            )  # average (mean here) of the 1000 samples appended to prediction list

            torch.cuda.empty_cache()
        benchmark_theta_pred_samples.append(time.time()-benchmark_start)

    theta_pred = torch.stack(theta_pred).cpu()  # back to cpu to convert to numpy

    benchmark_theta_pred_samples = np.array(benchmark_theta_pred_samples)
    benchmark_theta_pred_sampling = {
        "average sampling time": float(benchmark_theta_pred_samples.mean()),
        "max sampling time": float(benchmark_theta_pred_samples.max()),
        "min sampling time": float(benchmark_theta_pred_samples.min()),
        "total time sampling": float(benchmark_theta_pred_samples.sum()),
    },

    return theta_pred, benchmark_theta_pred_sampling

def model_metrics_benchmarking(best_model_info, theta_test_metrics, theta_pred, best_val_loss):
    total_time = time.time()
    per_bin_benchmarks = []
    best_model_info["metrics_per_bin"] = []
    for bin_index in range(3):
        per_bin_start = time.time()
        r2_bin   = r2_score(theta_test_metrics[:, bin_index], theta_pred[:, bin_index])
        rmse_bin = root_mean_squared_error(theta_test_metrics[:, bin_index], theta_pred[:, bin_index])
        mae_bin  = mean_absolute_error(theta_test_metrics[:, bin_index], theta_pred[:, bin_index])
        best_model_info["metrics_per_bin"].append((r2_bin, rmse_bin, mae_bin))
        per_bin_benchmarks.append(time.time()-per_bin_start)

    overall_start = time.time()
    r2   = r2_score(theta_test_metrics, theta_pred)
    rmse = root_mean_squared_error(theta_test_metrics, theta_pred)
    mae  = mean_absolute_error(theta_test_metrics, theta_pred)
    best_model_info["metrics"] = (r2, rmse, mae, best_val_loss)

    per_bin_benchmarks = np.array(per_bin_benchmarks)
    benchmark_model_metrics = {
            "per bin":{
                "average time": float(per_bin_benchmarks.mean()),
                "max time": float(per_bin_benchmarks.max()),
                "min time":float(per_bin_benchmarks.min()),
                "total time":float(per_bin_benchmarks.sum()),
            },
            "overall metrics time": float(time.time()-overall_start),
            "total metrics time":float(time.time()-total_time),
        }
    return best_model_info, r2, rmse, mae, benchmark_model_metrics

def reionization_and_pairplots_benchmarking():
    pass

def uncertainty_benchmarking():
    pass

def global_tarp_benchmarking(x_test, theta_test, best_posterior):
    benchmark_20_realizations = []
    def timeout_handler(signum, frame):
        raise TimeoutError("Global TARP timed out")

    global_tarp_curves = []
    mean_global_pval = 0.0
    successful_global_tarps = 0

    for k in range(20):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes

        try:
            benchmark_start = time.time()

            num_tarp_samples = 500
            tarp_index = torch.randperm(len(theta_test))[:num_tarp_samples]

            prior_samples = theta_test[tarp_index]
            prior_predictives = x_test[tarp_index]

            ecp, alpha = run_tarp(
                prior_samples,
                prior_predictives,
                best_posterior,
                references=None,
                num_posterior_samples=1000,
                use_batched_sampling=True,
            )

            _, ks_pval = check_tarp(ecp.cpu(), alpha.cpu())
            signal.alarm(0)
            benchmark_20_realizations.append(time.time()-benchmark_start)

            mean_global_pval += ks_pval
            successful_global_tarps += 1
            global_tarp_curves.append((ecp.cpu().numpy(), alpha.cpu().numpy()))

        except TimeoutError:
            print(f"Global TARP realization {k + 1} timed out, skipping...")
            continue

    if successful_global_tarps > 0:
        mean_global_pval /= successful_global_tarps

    benchmark_20_realizations = np.array(benchmark_20_realizations)
    benchmark_global_tarp = {
        "20 TARP average time for one realization": float(benchmark_20_realizations.mean()),
        "20 TARP max time": float(benchmark_20_realizations.max()),
        "20 TARP min time": float(benchmark_20_realizations.min()),
        "20 TARP total time": float(benchmark_20_realizations.sum()),
    },

    return mean_global_pval, global_tarp_curves, benchmark_global_tarp

def ppc_benchmarking(x_test, all_params, low_index, mid_index, high_index, scenario_samples):
    ppc_benchmark = []

    benchmark_total_start = time.time()
    x_test_np = x_test.cpu().numpy()
    theta_unique = all_params.cpu().numpy()

    nn_finder = NearestNeighbors(
        n_neighbors=1
    ).fit(
        theta_unique
    )  # we will be using the nearest theta instead of the simulator since we do not possess the latter

    n_ppc = 400  # we will be looking up 400 thetas, to ensure to get the nearest (not guaranteed unless n_ppc is higher)
    fig_ppc, axes_ppc = plt.subplots(1, 3, figsize=(20, 4))
    ppc_metrics = {}

    for ax, (name, index, color) in zip(
        axes_ppc,
        [
            ("low", low_index, "red"),
            ("mid", mid_index, "green"),
            ("high", high_index, "blue"),
        ],
    ):
        benchmark_start = time.time()
        x_obs = x_test_np[
            index
        ]  # index gets the observations that are linked in the specific scenario
        theta_p = scenario_samples[name][0][:n_ppc].numpy()  # shape (300, 3)

        distances, nn_index = nn_finder.kneighbors(theta_p)
        # (300, index observations)
        x_pp_list = []
        for index_pp in nn_index[:, 0]:
            start_index = index_pp * 20
            end_index = start_index + 20
            x_pp_list.append(x_test_np[start_index:end_index].mean(axis=0))
        x_pp = np.array(x_pp_list)
        x_pp_mean = x_pp.mean(axis=0)
        x_pp_std = x_pp.std(axis=0)

        frac_res = (x_obs - x_pp_mean) / (np.abs(x_pp_mean) + 1e-10)  # residulas
        z_res = (x_obs - x_pp_mean) / (x_pp_std + 1e-10)  # z score

        low_ppc = min(x_obs.min(), x_pp_mean.min())
        high_ppc = max(x_obs.max(), x_pp_mean.max())

        ppc_benchmark.append(time.time() - benchmark_start)

        ax.scatter(x_pp_mean, x_obs, color=color, alpha=0.3, s=5)
        ax.plot([low_ppc, high_ppc], [low_ppc, high_ppc], "k--", alpha=0.6, linewidth=1)
        ax.set_xlabel("mean(x_pp)")
        ax.set_ylabel("x_obs")

        ppc_metrics[name] = {
            "mean_abs_z": float(np.abs(z_res).mean()),
            "mean_abs_frac_residual": float(np.abs(frac_res).mean()),
            "mean_nn_dist": float(distances.mean()),
        }

        ax.set_title(
            f"{name}: mean |z|={ppc_metrics[name]['mean_abs_z']:.2f}"
            f"  mean |fractional residual|={ppc_metrics[name]['mean_abs_frac_residual']:.2f}"
        )

    ppc_benchmark = np.array(ppc_benchmark)
    benchmark_ppc = { #not including figure drawing
        "low PPC time": float(ppc_benchmark[0]),
        "mid PPC time": float(ppc_benchmark[1]),
        "high PPC time": float(ppc_benchmark[2]),
        "average PPC time": float(ppc_benchmark.mean()),
        "min PPC time": float(ppc_benchmark.min()),
        "max PPC time": float(ppc_benchmark.max()),
        "total PPC time": float(time.time()-benchmark_total_start), #including initialization
    },

    return ppc_metrics, fig_ppc, benchmark_ppc



#---
#utility
#---
def save_benchmark(pipeline_type, data_type, benchmarks):
    """
    benchmarks should follow the structure found in the bottom of this script
    """

    local_time = time.localtime() #to get day-hour-min

    if not os.path.isdir("benchmarks"):
        os.mkdir("benchmarks")

    with open(f"benchmarks/{pipeline_type}-{data_type}-benchmark-{local_time.tm_mday}-{local_time.tm_hour}-{local_time.tm_min}.json", "w") as f: #timestamp at the end in case we want multiple benchmarks of the same pipeline/data
        json.dump(
            benchmarks,
            f,
            indent=4,
        )

    pprint.pprint(benchmarks)

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
            "resnet": {
                "total resnet training time",
                "average resnet training time",
                "max resnet training time",
                "min resnet training time",
            },
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
            "sampling theta_pred":{
                "average sampling time",
                "max sampling time",
                "min sampling time",
                "total time sampling",
            },
            "model metrics":{
                "per bin":{
                    "average time",
                    "max time",
                    "min time",
                    "total time",
                },
                "overall metrics time",
                "total metrics time",
            },
            "global TARP":{
                "20 TARP min time",
                "20 TARP max time",
                "20 TARP average time for one realization",
                "20 TARP total time",
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
            "data loading time",
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
