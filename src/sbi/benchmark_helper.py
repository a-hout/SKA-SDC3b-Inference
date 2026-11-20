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
from compressor import ps2d_resnet
import torch
from mlflowadapter import MLflowAdapter
from sbi.neural_nets import posterior_nn
from torch.utils.data.dataset import Subset, random_split
from sbi.diagnostics import check_tarp, run_tarp
from sbi.inference import NPE
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sbi.analysis import pairplot
from scipy.stats import gaussian_kde
from tqdm import tqdm

#---
#model training
#---
def data_loading_benchmarking(gen, train_val_dataset, benchmark_data_loading, full_realization=True):
    """
    If full realization is set to false, then we randomly pick a realization TODO
    """
    benchmark_start = time.time()
    def split(dataset, full_realization=True):
        # get all items from the dataset in one pass this is C++ optimized in the backend
        # train_set[j] returns (param shape=(3,), obs shape=(20, 3, 3, 21) for betti)
        all_params, all_obs = zip(*[dataset[j] for j in range(len(dataset))])
        all_params = torch.stack(all_params)  # (N, 3)  ex -> (12000, 3)
        all_obs = torch.stack(all_obs)     # (N, 20, dot dot dot) ex -> (12000, 20, 3, 3, 21)

        #exxpand theta to match every noise realization
        N, n_noise = all_obs.shape[:2]  #N=12000, n_noise=20 (i think?)

        if full_realization:
            # unsquueeze adds a dimension so(12000, 3) -> (12000, 1, 3)
            # expand copies it 20 times so (12000, 1, 3) -> (12000, 20, 3)
            # reshpae flattens N*noise into one so (12000, 20, 3) -> (240000, 3) big array!
            all_params = all_params.unsqueeze(1).expand(N, n_noise, -1).reshape(N * n_noise, -1)

            #dlatten obs the same way
            # (12000, 20, 3, 3, 21) -> (240000, 3, 3, 21) (for betti, same logic for ps2d)
            all_obs = all_obs.reshape(N * n_noise, *all_obs.shape[2:])
        else:
            rand_int = torch.randint(n_noise, (N,), generator=gen)
            all_obs = all_obs[torch.arange(N), rand_int] #random realizations
        return all_params, all_obs

    train_set, test_set = random_split(
        train_val_dataset, [0.8, 0.2], generator=gen
    )
    theta_train, x_train = split(train_set, full_realization)
    theta_test, x_test = split(test_set, full_realization)

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

def resnet_benchmark(benchmark_embedding, data_type, x_train, x_test, theta_train, theta_test, summary_dim, get_compressor, ps2d_resnet, device):
    embedding = None

    batch_size = 512

    ### train a resnet model for the trial
    if data_type in ["ps2d", "ps2d+betti"]:
        benchmark_resnet_start = time.time()
        print("starting training for resnet!!!")

        if data_type == "ps2d+betti":
            x_train_img = x_train[:, :300].view(
                -1, 3, 10, 10
            )  # fit to number of samples and basic shape
            x_test_img = x_test[:, :300].view(
                -1, 3, 10, 10
            )  # fit to number of samples and basic shape
        else:
            x_train_img = x_train
            x_test_img = x_test
        resnet, epochs_resnet = ps2d_resnet.train_resnet(
            x_train_img,
            x_test_img,
            theta_train,
            theta_test,
            batch_size,
            device,
            benchmark=True
        )
        benchmark_embedding.append((time.time() - benchmark_resnet_start, epochs_resnet))
        print("finished resnet training")
        embedding = get_compressor(data_type, summary_dim, model=resnet)
    if embedding is None:
        embedding = get_compressor(data_type, summary_dim)

    return embedding, batch_size

def density_estimator_benchmark(benchmark_density_estimator_training, neural_posterior, prior, device, tracker, theta_train, x_train, batch_size, learning_rate):
    trainer = NPE(
        density_estimator=neural_posterior,
        prior=prior,
        device=device,
        tracker=tracker,  # built in by sbi for the tracker with adapter
    )

    benchmark_density_estimator_training_start = time.time()
    density_estimator = trainer.append_simulations(theta_train, x_train).train(
        training_batch_size=batch_size,
        show_train_summary=True,
        learning_rate=learning_rate,
    )
    benchmark_density_estimator_training.append((time.time()-benchmark_density_estimator_training_start, trainer.summary["epochs_trained"][-1]))
    return benchmark_density_estimator_training, trainer, density_estimator

def tarp_benchmarking(theta_test, x_test, posterior, benchmark_tarp_20_realizations: list):
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

def model_training_benchmark(BEST_PARAMS, mlflow, train_val_dataset, full_realization, get_compressor, data_type, device, prior, pipeline_type):
    benchmark_total_run_time_start = time.time()
    benchmark_training = []
    benchmark_data_loading = []
    benchmark_tarp_20_realizations = [] #contains all the times for 20*30 TARPS
    benchmark_embedding = []
    benchmark_density_estimator_training = []

    summary_dim, model, num_transforms, hidden_features, num_bins, learning_rate = (
        BEST_PARAMS.values()
    )

    model_info = {
        "number": [],
        "val_loss_values": [],
        "tarp_values": [],
        "posteriors": [],
        "density_estimators": [],
        "tarp_curves": [],
    }

    tarp_fig, tarp_axs = plt.subplots(
        6, 5, figsize=(20, 30)
    )  # 1 plot per one of the 30 models

    with mlflow.start_run(run_name="Best model run" if pipeline_type=="singular" else "Best ensemble model run"):
        for i in range(30):  # testing with 5, production with 30

            benchmark_training_time = time.time()

            model_info["number"].append(i + 1)
            gen = torch.manual_seed(i + 1)

            with mlflow.start_run(run_name=f"number_{i + 1}", nested=True):
                tracker = MLflowAdapter(mlflow)

                mlflow.log_params(
                    {
                        "number": i + 1,
                        "summary_dim": summary_dim,
                        "model": model,
                        "num_transforms": num_transforms,
                        "hidden_features": hidden_features,
                        "num_bins": num_bins,
                        "learning_rate": learning_rate,
                    }
                )

                theta_train, x_train, theta_test, x_test, benchmark_data_loading = data_loading_benchmarking(gen, train_val_dataset, benchmark_data_loading, full_realization)

                embedding, batch_size = resnet_benchmark(benchmark_embedding, data_type, x_train, x_test, theta_train, theta_test, summary_dim, get_compressor, ps2d_resnet, device)

                neural_posterior = posterior_nn(
                    model=model,
                    embedding_net=embedding,
                    z_score_x="none" if data_type != "betti" else "independent", #we train on raw obs the resnet for ps2d, dont want to mix and match
                    hidden_features=hidden_features,
                    num_transforms=num_transforms,
                    num_bins=num_bins,
                )

                benchmark_density_estimator_training, trainer, density_estimator = density_estimator_benchmark(benchmark_density_estimator_training, neural_posterior, prior, device, tracker, theta_train, x_train, batch_size, learning_rate)

                posterior = trainer.build_posterior(density_estimator)

                val_loss = trainer._summary["best_validation_loss"][0]
                model_info["val_loss_values"].append((i + 1, val_loss))
                mlflow.log_metric("val_loss", float(val_loss))

                # tarp
                mean_pval_tarp, current_model_curves, benchmark_tarp_20_realizations = tarp_benchmarking(theta_test, x_test, posterior, benchmark_tarp_20_realizations)

                model_info["tarp_values"].append((i + 1, mean_pval_tarp))
                model_info["tarp_curves"].append((i + 1, current_model_curves))

                mlflow.log_metrics({"tarp_pval_mean": float(mean_pval_tarp)})

                model_info["density_estimators"].append((i + 1, density_estimator))
                model_info["posteriors"].append((i + 1, posterior))

                del trainer, density_estimator, neural_posterior, embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            benchmark_training.append(time.time()-benchmark_training_time)

        if pipeline_type=="singular":
            # best model is model with best mean tarp
            tarp_sorted = sorted(model_info["tarp_values"], key=lambda x: x[1], reverse=True)
            loss_dict = dict(model_info["val_loss_values"])
            best_num = tarp_sorted[0][0]
            best_val_loss = loss_dict[best_num]

            print(f"best model -> trial {best_num}")

            # drawing on tarp_fig the 30 models tarps and their 20 realisations, in green the best model
            curves_dict = dict(model_info["tarp_curves"])
            pval_dict = dict(model_info["tarp_values"])

            for index, ax in enumerate(tarp_axs.ravel()):
                run_num = index + 1

                if run_num not in curves_dict:
                    ax.axis("off")  # remove trial which have no complete realisations
                    continue

                line_color = "green" if run_num == best_num else "red"

                ax.plot(
                    [0, 1], [0, 1], color="black", linestyle="--", alpha=0.8, zorder=3
                )  # diagonal reference line in background

                for ecp_vals, alpha_vals in curves_dict[run_num]:
                    ax.plot(
                        alpha_vals, ecp_vals, color=line_color, alpha=0.15, zorder=2
                    )  # plotting all realsiations

                ax.set_title(f"trial {run_num}: p={pval_dict[run_num]:.3f}")
                ax.set_xlabel("Credibility level")
                if (
                    index == 0 or index % 5 == 0
                ):  # only show y label on leftmost plots, thanks chatgpt
                    ax.set_ylabel("Expected coverage probability")

            tarp_fig.savefig(f"models/sbi/{data_type}/tarp_plot.png")
            mlflow.log_artifact(f"models/sbi/{data_type}/tarp_plot.png")

            mlflow.log_metric("best model number", best_num)
            pipeline_specific = (best_num, best_val_loss)
        else: #for ensemble
            # best model is 5 best models with best mean tarp
            posteriors = dict(model_info["posteriors"])
            tarp_sorted = sorted(model_info["tarp_values"], key=lambda x: x[1], reverse=True)
            best_posteriors = []
            selected_runs = []

            for run_num, p_val in tarp_sorted:  # get 5 vest calibrated models
                if len(best_posteriors) >= 5:
                    break
                if run_num in posteriors:
                    best_posteriors.append(posteriors[run_num])
                    selected_runs.append(run_num)

                    mlflow.log_metric(f"ensemble_member_{len(best_posteriors)}", run_num)
            print(f"Selected ensemble trials: {selected_runs}")

            loss_dict = dict(model_info["val_loss_values"])
            curves_dict = dict(model_info["tarp_curves"])
            pval_dict = dict(model_info["tarp_values"])

            for index, ax in enumerate(tarp_axs.ravel()):
                run_num = index + 1

                if run_num not in curves_dict:
                    ax.axis("off")
                    continue

                line_color = "green" if run_num in selected_runs else "red"

                ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.8, zorder=3)

                for ecp_vals, alpha_vals in curves_dict[run_num]:
                    ax.plot(alpha_vals, ecp_vals, color=line_color, alpha=0.15, zorder=2)

                ax.set_title(f"trial {run_num}: p={pval_dict[run_num]:.3f}")

                ax.set_xlabel("Credibility level")
                if index == 0 or index % 5 == 0:
                    ax.set_ylabel("Expected coverage probability")

            tarp_fig.savefig(f"models/ensemble-sbi/{data_type}/tarp_plot.png")
            mlflow.log_artifact(f"models/ensemble-sbi/{data_type}/tarp_plot.png")

            # log primary best tracker (the absolute best model)
            if selected_runs:
                best_val_loss = loss_dict[selected_runs[0]]
                mlflow.log_metric("best model number", selected_runs[0])
            pipeline_specific = (best_posteriors, best_val_loss)

    benchmark_total_run_time = time.time() - benchmark_total_run_time_start
    data_loading_benchmark = data_loading_benchmark_measures(benchmark_data_loading)
    tarp_benchmark = tarp_benchmark_measures(benchmark_tarp_20_realizations)
    benchmark_training = np.array(benchmark_training)
    times, epochs = np.array(benchmark_density_estimator_training).T
    model_training_benchmark = {
        "training time": {
            "total run time": float(benchmark_total_run_time), #including drawing figures and finding the best model
            "total training time": float(benchmark_training.sum()), #not including the above
            "average training time": float(benchmark_training.mean()),
            "max training time": float(benchmark_training.max()),
            "min training time": float(benchmark_training.min()),
        },
        "data loading": data_loading_benchmark,
        "density estimator training":{ #not including data loading, embedding, tarp, etc.
            "average training time": float(times.mean()),
            "max training time": float(times.max()),
            "min training time": float(times.min()),
            "average training time per epoch": float((times/epochs).mean())
        },
        "TARP": tarp_benchmark,
    }
    if data_type in ["ps2d", "ps2d+betti"]:
        resnet_times, resnet_epochs = np.array(benchmark_embedding).T
        model_training_benchmark["resnet"] = {
            "total resnet training time": float(resnet_times.sum()),
            "average resnet training time": float(resnet_times.mean()),
            "max resnet training time": float(resnet_times.max()),
            "min resnet training time": float(resnet_times.min()),
            "average training time per epoch": float((resnet_times/resnet_epochs).mean())
        },

    return model_training_benchmark, model_info, pipeline_specific
#---
#model testing
#---

def test_data_loading_benchmark(global_test_dataset, full_realization=True):
    benchmark_test_data = time.time()
    #if full_realization:
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

def model_testing_benchmark(mlflow, global_test_dataset, full_realization, best_posterior, best_val_loss, data_type, pipeline_type, best_posteriors=None):
    best_model_info = {}
    with mlflow.start_run(run_name="Best model evaluation"if pipeline_type=="singular" else "Best ensemble model evaluation"):
        tracker = MLflowAdapter(mlflow)

        benchmark_testing_start = time.time()

        all_params, x_test, theta_test, n_noise, benchmark_test_data_loading = test_data_loading_benchmark(global_test_dataset, full_realization)

        theta_test_cpu = theta_test.cpu()

        # metrics
        metrics_indices = list(range(0, len(x_test), n_noise))
        x_test_metrics = x_test[metrics_indices]
        theta_test_metrics = theta_test_cpu[metrics_indices]

        theta_pred, benchmark_theta_pred_sampling = theta_pred_sampling_benchmark(x_test_metrics, best_posterior)

        best_model_info, r2, rmse, mae, benchmark_model_metrics = model_metrics_benchmarking(best_model_info, theta_test_metrics, theta_pred, best_val_loss)

        # scenarios (low = fast, high = slow reionization)
        theta_mean = theta_test.cpu().mean(dim=1)  # mean across bins for each observation
        low_target = torch.quantile(theta_mean, 0.16)  # 16th percentile, not extreme at 0
        mid_target = torch.median(theta_mean)  # 50th percentile
        high_target = torch.quantile(
            theta_mean, 0.84
        )  # 84th percentile, not extreme at 100

        # get the indices of the 3 sceanrios
        low_index = (theta_mean - low_target).abs().argmin().item()
        mid_index = (theta_mean - mid_target).abs().argmin().item()
        high_index = (theta_mean - high_target).abs().argmin().item()

        print(
            f"Indices of scenarios:\n\tLow: {theta_mean[low_index]:.3f}\n\tmid: {theta_mean[mid_index]:.3f}\n\thigh: {theta_mean[high_index]:.3f}"
        )

        z_centers = [158.45, 173.45, 188.45]
        r_range = ["151.0-165.9", "166.0-180.9", "181.0-195.9"]

        scenario_samples = {}

        pairplot_figs = {}
        check_figs = {}
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))

        for name, index, color in [
            ("low", low_index, "red"),
            ("mid", mid_index, "green"),
            ("high", high_index, "blue"),
        ]:
            with torch.no_grad():
                if pipeline_type == "ensemble":
                    ensemble_samples = [
                        posterior.sample(
                            (1000,), x=x_test[index], show_progress_bars=False
                        ).cpu()
                        for posterior in best_posteriors
                    ]
                samples = best_posterior.sample(
                    (5000,),
                    x=x_test[index],
                    show_progress_bars=False,
                ).cpu()
                scenario_samples[name] = (
                    samples,
                    index,
                    color,
                    theta_test[index].cpu().numpy(),
                )  # avoid resampling later
            torch.cuda.empty_cache()

            x_z = samples.mean(dim=0).numpy()  # mean of distribution
            lower = samples.quantile(0.16, dim=0).numpy()  # -1sigma from distribution
            upper = samples.quantile(0.84, dim=0).numpy()  # +1sigma
            true_theta = theta_test[index].cpu().numpy()

            for i, zbin in enumerate(r_range):
                mlflow.log_metrics(
                    {
                        f"xHI_mean_{name}_z{i + 1}": float(x_z[i]),
                        f"xHI_lower_{name}_z{i + 1}": float(lower[i]),
                        f"xHI_upper_{name}_z{i + 1}": float(upper[i]),
                    }
                )

            # plto reionisation fraction history
            axes.plot(z_centers, x_z, "o-", color=color, label=f"{name} posterior xHI")
            axes.fill_between(z_centers, lower, upper, color=color, alpha=0.2)
            axes.plot(
                z_centers,
                true_theta,
                "x--",
                color=color,
                markersize=10,
                alpha=0.7,
                label=f"{name} true xHI",
            )

            fig_pair, ax = pairplot(
                samples,
                points=true_theta.reshape(1, -1),
                labels=["xHI_z1", "xHI_z2", "xHI_z3"],
                fig_kwargs={
                    "points_colors": ["red"],
                },
            )
            pairplot_figs[name] = fig_pair
            fig_pair.savefig(
                f"models/{'sbi' if pipeline_type=='singular' else 'ensemble-sbi'}/{data_type}/pairplot_{name}_xHI.png"
            )  # also available via mlflow
            if pipeline_type=='ensemble':
                # plot the posteriors
                fig_check, axes_check = pairplot(
                    ensemble_samples,
                    points=true_theta.reshape(1, -1),
                    labels=["xHI_z1", "xHI_z2", "xHI_z3"],
                    fig_kwargs={
                        "points_colors": ["red"],
                    },
                )
                fig_check.savefig(
                    f"models/ensemble-sbi/{data_type}/ensemble_check_pairplot_{name}.png"
                )
                check_figs[name] = fig_check

        for boundary in [165.9, 180.9]:
            axes.axvline(boundary, color="gray", linestyle="--", alpha=0.5)

        axes.set_xlabel("Frequency (MHz)")
        axes.set_ylabel("mean_xHI")
        axes.set_title("Reionization History x(z)")
        axes.legend()

        fig.savefig(f"models/{'sbi' if pipeline_type=='singular' else 'ensemble-sbi'}/{data_type}/reionization_history.png")

        # uncetainty P(x|z) plots
        fig_pxz, axes_pxz = plt.subplots(
            1, 3, figsize=(15, 4), sharey=True
        )  # three plots sharing a y

        for ax, (name, index, color) in zip(
            axes_pxz,
            [
                ("low", low_index, "red"),
                ("mid", mid_index, "green"),
                ("high", high_index, "blue"),
            ],
        ):
            samples = scenario_samples[name][0]

            true_theta = scenario_samples[name][3]

            x_range = np.linspace(0, 1, 200)
            for i, (zbin, ls) in enumerate(zip(r_range, ["-", "--", ":"])):
                kde = gaussian_kde(samples[:, i].numpy())
                ax.plot(x_range, kde(x_range), color=color, linestyle=ls, label=zbin)
                # ground truth vertical line
                ax.axvline(true_theta[i], color=color, linestyle=ls, alpha=0.5, linewidth=2)

            ax.set_title(f"{name} xHI")
            ax.set_xlabel("mean_xHI")
            ax.legend(fontsize=7)

        axes_pxz[0].set_ylabel("P(x|z)")
        fig_pxz.savefig(f"models/{'sbi' if pipeline_type=='singular' else 'ensemble-sbi'}/{data_type}/pxz_plot.png")

        # tarp plot redo, since new test data
        mean_global_pval, global_tarp_curves, benchmark_global_tarp = global_tarp_benchmarking(x_test, theta_test, best_posterior)

        # get final global TARP plot
        tarp_eval_fig, tarp_eval_ax = plt.subplots(figsize=(6, 6))
        tarp_eval_ax.plot(
            [0, 1], [0, 1], color="black", linestyle="--", alpha=0.8, label="Ideal"
        )

        for ecp_vals, alpha_vals in global_tarp_curves:
            tarp_eval_ax.plot(alpha_vals, ecp_vals, color="purple", alpha=0.15, linewidth=1)

        tarp_eval_ax.set_title(f"Global test set TARP: p = {mean_global_pval:.6f}")
        tarp_eval_ax.set_xlabel("Credibility level")
        tarp_eval_ax.set_ylabel("Expected coverage probability")

        tarp_eval_fig.savefig(f"models/{'sbi' if pipeline_type=='singular' else 'ensemble-sbi'}/{data_type}/global_tarp_plot.png")
        tracker.add_figure("global_tarp_plot", tarp_eval_fig)

        # PPC
        ppc_metrics, fig_ppc, benchmark_ppc = ppc_benchmarking(x_test, all_params, low_index, mid_index, high_index, scenario_samples)

        fig_ppc.suptitle("Posterior predictive check (KNN): x_obs vs x_pp")
        fig_ppc.savefig(f"models/{'sbi' if pipeline_type=='singular' else 'ensemble-sbi'}/{data_type}/ppc_plot.png")
        tracker.add_figure("ppc_plot", fig_ppc)
        for name, m in ppc_metrics.items():
            mlflow.log_metrics(
                {
                    f"ppc_mean_z_{name}": m["mean_abs_z"],
                    f"ppc_frac_res_{name}": m["mean_abs_frac_residual"],
                    f"ppc_nn_dist_{name}": m["mean_nn_dist"],
                }
            )

        # save evaluation
        with open(f"models/{'sbi' if pipeline_type=='singular' else 'ensemble-sbi'}/{data_type}/evaluation.json", "w") as f:
            json.dump(
                {
                    "r2": float(best_model_info["metrics"][0]),
                    "rmse": float(best_model_info["metrics"][1]),
                    "mae": float(best_model_info["metrics"][2]),
                    "bin_metrics": {
                        r_range[i]: {
                            "r2": float(best_model_info["metrics_per_bin"][i][0]),
                            "rmse": float(best_model_info["metrics_per_bin"][i][1]),
                            "mae": float(best_model_info["metrics_per_bin"][i][2]),
                        }
                        for i in range(3)
                    },
                    "global_tarp_ks_pval": float(mean_global_pval),
                    "val_loss": float(best_val_loss),
                    "ppc_metrics": ppc_metrics,
                },
                f,
                indent=4,
            )

        mlflow.log_metrics(
            {
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "global_tarp_ks_pval": float(mean_global_pval),
            }
        )
        # log per-bin metrics
        for i, zbin in enumerate(r_range):
            mlflow.log_metrics(
                {
                    f"r2_{zbin}": float(best_model_info["metrics_per_bin"][i][0]),
                    f"rmse_{zbin}": float(best_model_info["metrics_per_bin"][i][1]),
                    f"mae_{zbin}": float(best_model_info["metrics_per_bin"][i][2]),
                }
            )

        # log figures
        tracker.add_figure("reionization_history", fig)
        tracker.add_figure("pxz_plot", fig_pxz)
        for name in ["low", "mid", "high"]:
            tracker.add_figure(f"pairplot_{name}_xHI", pairplot_figs[name])
            if pipeline_type=="ensemble":
                tracker.add_figure(f"ensemble_check_pairplot_{name}", check_figs[name])

        model_testing_benchmark = {
            "sampling theta_pred":benchmark_theta_pred_sampling,
            "model metrics": benchmark_model_metrics,
            "global TARP": benchmark_global_tarp,
            "PPC": benchmark_ppc,
            "data loading time": float(benchmark_test_data_loading),
            "total testing time": float(time.time() - benchmark_testing_start),
        }

    mlflow.end_run()
    return model_testing_benchmark

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
