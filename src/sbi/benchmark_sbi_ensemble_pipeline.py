"""
similar to the sbi_ensemble_pipeline.py file but specific to benchmarking
"""

##################################################################################################################
# 0. imports
##################
import json
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import benchmark_helper
import torch
from compressor import (
    betti_compressor,
    joint_ps2d_betti_compressor,
    ps2d_resnet,
)
from dataloader import (
    data_type_loader,
    joint_ps2d_betti_loader,
)
from mlflowadapter import MLflowAdapter
from sbi.analysis import pairplot
from sbi.inference import NPE, EnsemblePosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset, random_split
import time

##################
# 1. inits
##################

# time benchmarking inits
benchmarks = {}
benchmark_total_pipeline_time_start = time.time()

pipeline_type = "ensemble"
full_realization = True #true if all 20 realization, else we should be picking a random realization in dataloaders

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
        "Normal usage is: python src/sbi/benchmark_sbi_ensemble_pipeline.py <data_type> <best params path>"
    )
    exit(1)

try:
    with open(sys.argv[2]) as f:  # load path with best model parameters
        BEST_PARAMS = json.load(f)["best_params"]
except (IndexError, FileNotFoundError):
    print(
        "Normal usage is: python src/sbi/benchmark_sbi_ensemble_pipeline.py <data_type> <best params path>"
    )
    exit(1)

dataset = None

match data_type:
    case "ps2d":
        mlflow.set_experiment("SBI Ensemble PS2D")

    case "betti":
        mlflow.set_experiment("SBI Ensemble Betti")

    case "ps2d+betti" | "betti+ps2d":  # combine both ps2d and betti
        data_type = "ps2d+betti"
        mlflow.set_experiment("SBI Ensemble PS2D & Betti")

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
prior = BoxUniform(
    low=torch.zeros(3), high=torch.ones(3), device=device
)  # ensembles can be finicky with their posteriors, so widening to 0, 1 is a sensible choice and any deviations can be rapidly observed during evaluation
# prior = BoxUniform(low=low, high=high, device=device)

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
    (Subset(dataset, range(int(len(dataset))))), #when testing during demo since it's more rapid
    [0.8, 0.2],
    generator=global_gen,
)

##################
# 2. 30-run
##################
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

with mlflow.start_run(run_name="Best ensemble model run"):
    for i in range(30):
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

            theta_train, x_train, theta_test, x_test, benchmark_data_loading = benchmark_helper.data_loading_benchmark(gen, train_val_dataset, benchmark_data_loading, full_realization)

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
                resnet = ps2d_resnet.train_resnet(
                    x_train_img,
                    x_test_img,
                    theta_train,
                    theta_test,
                    batch_size,
                    device,
                )
                benchmark_embedding.append(time.time() - benchmark_resnet_start)
                print("finished resnet training")
                embedding = get_compressor(data_type, summary_dim, model=resnet)
            if embedding is None:
                embedding = get_compressor(data_type, summary_dim)
            neural_posterior = posterior_nn(
                model=model,
                embedding_net=embedding,
                z_score_x="none" if data_type != "betti" else "independent", #we train on raw obs the resnet for ps2d, dont want to mix and match
                hidden_features=hidden_features,
                num_transforms=num_transforms,
                num_bins=num_bins,
            )

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

            posterior = trainer.build_posterior(density_estimator)

            val_loss = trainer._summary["best_validation_loss"][0]
            model_info["val_loss_values"].append((i + 1, val_loss))
            mlflow.log_metric("val_loss", float(val_loss))

            # tarp
            mean_pval_tarp, current_model_curves, benchmark_tarp_20_realizations = benchmark_helper.tarp_benchmark(theta_test, x_test, posterior, benchmark_tarp_20_realizations)

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


benchmark_total_run_time = time.time() - benchmark_total_run_time_start
data_loading_benchmark = benchmark_helper.data_loading_benchmark_measures(benchmark_data_loading)
tarp_benchmark = benchmark_helper.tarp_benchmark_measures(benchmark_tarp_20_realizations)
benchmark_training = np.array(benchmark_training)
benchmark_embedding = np.array(benchmark_embedding)
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
    model_training_benchmark["resnet"] = {
        "total resnet training time": float(benchmark_embedding.sum()),
        "average resnet training time": float(benchmark_embedding.mean()),
        "max resnet training time": float(benchmark_embedding.max()),
        "min resnet training time": float(benchmark_embedding.min()),
    },
print(model_training_benchmark)

##################
# 3. best ensemble model
##################
best_posterior = EnsemblePosterior(best_posteriors, device=device)
torch.save(best_posterior, f"models/ensemble-sbi/{data_type}/best_posterior.pt")

##################
# 5. eval
##################
best_model_info = {}
with mlflow.start_run(run_name="Best ensemble model evaluation"):
    tracker = MLflowAdapter(mlflow)

    benchmark_testing_start = time.time()

    all_params, x_test, theta_test, n_noise, benchmark_test_data_loading = benchmark_helper.test_data_loading_benchmark(global_test_dataset, full_realization)

    theta_test_cpu = theta_test.cpu()

    # metrics
    metrics_indices = list(range(0, len(x_test), n_noise))
    x_test_metrics = x_test[metrics_indices]
    theta_test_metrics = theta_test_cpu[metrics_indices]

    theta_pred, benchmark_theta_pred_sampling = benchmark_helper.theta_pred_sampling_benchmark(x_test_metrics, best_posterior)

    best_model_info, r2, rmse, mae, benchmark_model_metrics = benchmark_helper.model_metrics_benchmarking(best_model_info, theta_test_metrics, theta_pred, best_val_loss)

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
            "--",
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
            f"models/ensemble-sbi/{data_type}/pairplot_{name}_xHI.png"
        )  # also available via mlflow

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

    fig.savefig(f"models/ensemble-sbi/{data_type}/reionization_history.png")

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
    fig_pxz.savefig(f"models/ensemble-sbi/{data_type}/pxz_plot.png")

    # tarp plot redo, since new test data
    mean_global_pval, global_tarp_curves, benchmark_global_tarp = benchmark_helper.global_tarp_benchmarking(x_test, theta_test, best_posterior)

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

    tarp_eval_fig.savefig(f"models/ensemble-sbi/{data_type}/global_tarp_plot.png")
    tracker.add_figure("global_tarp_plot", tarp_eval_fig)

    # PPC
    ppc_metrics, fig_ppc, benchmark_ppc = benchmark_helper.ppc_benchmarking(x_test, all_params, low_index, mid_index, high_index, scenario_samples)

    fig_ppc.suptitle("Posterior predictive check (KNN): x_obs vs x_pp")
    fig_ppc.savefig(f"models/ensemble-sbi/{data_type}/ppc_plot.png")
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
    with open(f"models/ensemble-sbi/{data_type}/evaluation.json", "w") as f:
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

#save benchmark results
benchmarks["benchmarks"] = {
    "model training": model_training_benchmark,
    "model testing": model_testing_benchmark,
    "total pipeline time":float(time.time() - benchmark_total_pipeline_time_start)
}
benchmark_helper.save_benchmark(pipeline_type, data_type, benchmarks)
