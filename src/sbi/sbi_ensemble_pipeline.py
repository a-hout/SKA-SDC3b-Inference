##################################################################################################################

"""
Ensemble pipeline training, similar to the standard sbi pipeline

1. Init of parameters and data

2. Train 30 times with different seeds to validate stability
train -> get val-loss -> run tarp (with 10 min timeout) and get calibration results

3. at the end, get the 5 best models with the best calibration (most negative val-loss and tarp closest to 1) and create an Ensemble Posterior from it

4. Full evaluation with metrics, overal and per z_x
R2
RMSE
MAE
Val loss

pairplots

reionisation history with the three scenarios + the actual mean ground truth for comparison

P(x | z) plot with the three scenarios (3 plots) (and comparison with ground truth?)
"""

##################################################################################################################
# 0. imports
##################
import json
import os
import signal
import sys
from itertools import product

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from compressor import betti_compressor, joint_ps2d_betti_compressor, ps2d_compressor
from dataloader import betti_loader, joint_ps2d_betti_loader, ps2d_loader
from mlflowadapter import MLflowAdapter
from sbi.analysis import pairplot
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics import check_tarp, run_tarp
from sbi.inference import NPE, EnsemblePosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset, random_split
from tqdm import tqdm

##################
# 1. inits
##################

device = (
    "cuda:3"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
_ = torch.manual_seed(26)

try:
    data_type = sys.argv[1].lower().strip("")
except IndexError:
    print(
        "Normal usage is: python src/sbi/sbi_ensemble_pipeline.py <data_type> <best params path>"
    )
    exit(1)

try:
    with open(sys.argv[2]) as f:  # load path with best model parameters
        BEST_PARAMS = json.load(f)["best_params"]
except (IndexError, FileNotFoundError):
    print(
        "Normal usage is: python src/sbi/sbi_ensemble_pipeline.py <data_type> <best params path>"
    )
    exit(1)

match data_type:
    case "ps2d":
        mlflow.set_experiment("Ensemble SBI PS2D")
        # embedding = ps2d_compressor.PS2DCompressor(3)

        dataset = ps2d_loader.PS2DDataset(
            "datasets/pk_128_train_250325_noise/memmap/", device=device
        )

        theta, _ = dataset[:]
        low = theta.min(dim=0).values.detach().clone().to(device)
        high = theta.max(dim=0).values.detach().clone().to(device)

    case "betti":
        mlflow.set_experiment("Ensemble SBI Betti")
        # embedding = betti_compressor.BettiCompressor(3)

        dataset = betti_loader.BettiDataset("datasets/betti/", device=device)

        theta = torch.tensor(dataset.params[:, 0, :], device=dataset.device)

        low = theta.min(dim=0).values.detach().clone()
        high = theta.max(dim=0).values.detach().clone()

    case "wst":
        mlflow.set_experiment("Ensemble SBI WST")

        pass

    case "ps2d+betti" | "betti+ps2d":  # combine both ps2d and betti
        data_type = "ps2d+betti"
        mlflow.set_experiment("SBI PS2D & Betti")

        # we need to see the indices for which ps2d and betti have matching thetas
        MEMMAP_PATH_PS2D = "datasets/pk_128_train_250325_noise/memmap/"
        MEMMAP_PATH_BETTI = "datasets/betti/"
        betti_theta = np.load(MEMMAP_PATH_BETTI + "sbi2_y.npy").astype(np.float32)
        ps2d_theta = np.memmap(
            MEMMAP_PATH_PS2D + "theta_memmap.npy",
            shape=(159449, 3),
            mode="r",
            dtype=np.float32,
        )

        PRECISION_DECIMALS = 5  # if we don't round, then only a few matching thetas but we also dont want to round too much
        betti_rounded = np.round(betti_theta, decimals=PRECISION_DECIMALS)
        ps2d_rounded = np.round(ps2d_theta, decimals=PRECISION_DECIMALS)

        betti_dict = {
            tuple(map(float, row)): index
            for index, row in enumerate(betti_rounded[:, 0, :])
        }  # because shape (15951, 20, 3)

        valid_pairs = []

        for index, row in enumerate(ps2d_rounded):
            row_tuple = tuple(map(float, row.ravel()))
            if row_tuple in betti_dict:
                valid_pairs.append((index, betti_dict[row_tuple]))

        ps2d_dataset = ps2d_loader.PS2DDataset(MEMMAP_PATH_PS2D, device=device)
        betti_dataset = betti_loader.BettiDataset(MEMMAP_PATH_BETTI, device=device)

        dataset = joint_ps2d_betti_loader.JointPS2DBettiDataset(
            ps2d_dataset, betti_dataset, valid_pairs
        )

        theta, _ = ps2d_dataset[:]  # because thetas are from ps2d
        low = theta.min(dim=0).values.detach().clone().to(device)
        high = theta.max(dim=0).values.detach().clone().to(device)

    case "ps2d+wst" | "wst+ps2d":  # combine both ps2d and wst
        data_type = "ps2d+wst"
        mlflow.set_experiment("Ensemble SBI PS2D & WST")
        pass

    case "betti+wst" | "wst+betti":  # combine both betti and wst
        data_type = "betti+wst"
        mlflow.set_experiment("Ensemble SBI Betti & WST")

        pass

    case (  # combine all data types
        "ps2d+betti+wst"
        | "ps2d+wst+betti"
        | "betti+ps2d+wst"
        | "betti+wst+ps2d"
        | "wst+ps2d+betti"
        | "wst+betti+ps2d"
    ):
        data_type = "ps2d+betti+wst"
        mlflow.set_experiment("SBI PS2D & Betti & WST")

    case other:
        print("No such data type or data type combination.")
        exit(1)


def get_compressor(data_type, summary_dim):
    match data_type:
        case "ps2d":
            return ps2d_compressor.PS2DCompressor(summary_dim=summary_dim)
        case "betti":
            return betti_compressor.BettiCompressor(summary_dim=summary_dim)
        case "ps2d+betti":
            return joint_ps2d_betti_compressor.JointPS2DBettiCompressor(
                ps2d_compressor.PS2DCompressor(summary_dim=summary_dim),
                betti_compressor.BettiCompressor(summary_dim=summary_dim),
                summary_dim=summary_dim,
            )
        # case other:
        # pass  # do later for wst and for combinations


prior = BoxUniform(low=low, high=high, device=device)

##################
# 2. 30-run
##################
summary_dim, model, num_transforms, hidden_features, num_bins, learning_rate = (
    BEST_PARAMS.values()
)
global_gen = torch.manual_seed(26)
train_val_dataset, global_test_dataset = random_split(
    dataset, [0.8, 0.2], generator=global_gen
)
# train_val_dataset = Subset(train_val_dataset, range(500))  # for testing
model_info = {
    "number": [],
    "val_loss_values": [],
    "tarp_values": [],
    "tarp_atc_values": [],
    "posteriors": [],
    "density_estimators": [],
}

with mlflow.start_run(run_name="Best ensemble model run"):
    for i in range(30):
        try:
            print("making model directory...")
            os.mkdir(f"models/ensemble-sbi/{data_type}/trial-{i + 1}/")
        except OSError:
            print("directory already exists.")

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

            train_set, test_set = random_split(
                train_val_dataset, [0.8, 0.2], generator=gen
            )

            theta_train, x_train = zip(*[train_set[j] for j in range(len(train_set))])
            theta_train = torch.stack(theta_train)
            # theta_train = (theta_train - theta_low) / (theta_high - theta_low + 1e-8)
            x_train = torch.stack(x_train)

            theta_test, x_test = zip(*[test_set[j] for j in range(len(test_set))])
            theta_test = torch.stack(theta_test)
            # theta_test = (theta_test - theta_low) / (theta_high - theta_low + 1e-8)
            x_test = torch.stack(x_test)

            neural_posterior = posterior_nn(
                model=model,
                embedding_net=get_compressor(data_type, summary_dim),
                # z_score_x="none",
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

            density_estimator = trainer.append_simulations(theta_train, x_train).train(
                training_batch_size=512,
                show_train_summary=True,
                learning_rate=learning_rate,
            )

            posterior = trainer.build_posterior(density_estimator)

            val_loss = trainer._summary["best_validation_loss"][0]
            model_info["val_loss_values"].append((i + 1, val_loss))
            mlflow.log_metric("val_loss", float(val_loss))

            # tarp
            def timeout_handler(signum, frame):
                """
                we need to kill the TARPing if it's >10 minutes (bad luck due to bad posterior/prior sampling)
                """
                raise TimeoutError("TARP timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10 minutes

            try:
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

                atc, ks_pval = check_tarp(ecp.cpu(), alpha.cpu())
                signal.alarm(0)
                model_info["tarp_values"].append((i + 1, ks_pval))
                model_info["tarp_atc_values"].append((i + 1, atc))
                tarp_fig, _ = plot_tarp(ecp.cpu(), alpha.cpu())
                tarp_fig.savefig(
                    f"models/ensemble-sbi/{data_type}/trial-{i + 1}/tarp_plot.png"
                )
            except TimeoutError:
                exit(0)  # continue  # go to the next run

            mlflow.log_metrics(
                {
                    "tarp_pval": float(ks_pval),
                    "tarp_atc": float(
                        atc
                    ),  # not used in the scoring but can be interesting to observe
                }
            )

            tracker.add_figure("tarp_plot", tarp_fig)

            model_info["density_estimators"].append((i + 1, density_estimator))
            model_info["posteriors"].append((i + 1, posterior))

    posteriors = dict(model_info["posteriors"])
    tarp_dict = dict(model_info["tarp_values"])  # this way we have {number: ks_pval}

    loss_sorted = sorted(model_info["val_loss_values"], key=lambda x: x[1])

    best_posteriors = []
    selected_runs = set()
    for k in range(5):  # get 5 best models
        # best model is first model that is also well calbirated

        best_num = None
        best_val_loss = float("inf")  # infinity is bad

        for run_num, val_loss in loss_sorted:
            if run_num in selected_runs:
                continue
            if run_num not in tarp_dict:
                continue  # means tarp could not be ran for this model
            if tarp_dict[run_num] >= 0.99:
                best_num = run_num

                break  # get first best model with good calibration
        if best_num is None:
            for run_num, val_loss in loss_sorted:
                if run_num not in selected_runs:
                    best_num = run_num
                    break

        if best_num is not None:
            best_posteriors.append(posteriors[best_num])
            selected_runs.add(best_num)
        else:
            break

        mlflow.log_metric("best model number", best_num)


tarp_dict_atc = dict(model_info["tarp_atc_values"])

##################
# 3. best ensemble model
##################
best_posterior = EnsemblePosterior(best_posteriors, device=device)
torch.save(best_posterior, f"models/ensemble-sbi/{data_type}/best_posterior.pt")

##################
# 4. eval
##################
best_model_info = {}
with mlflow.start_run(run_name="best_model_evaluation"):
    tracker = MLflowAdapter(mlflow)
    gen = torch.manual_seed(26)

    theta_test, x_test = zip(
        *[global_test_dataset[i] for i in range(len(global_test_dataset))]
    )
    theta_test = torch.stack(theta_test)
    x_test = torch.stack(x_test)

    # metrics
    theta_pred = []
    for x_obs in tqdm(x_test, desc="Sampling test observations"):
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

    theta_pred = torch.stack(theta_pred).cpu()  # back to cpu to convert to numpy
    theta_test_cpu = theta_test.cpu()

    best_model_info["metrics_per_bin"] = []
    for bin_index in range(3):  # getting metrics per redshift range
        r2_bin = r2_score(theta_test_cpu[:, bin_index], theta_pred[:, bin_index])
        rmse_bin = root_mean_squared_error(
            theta_test_cpu[:, bin_index], theta_pred[:, bin_index]
        )
        mae_bin = mean_absolute_error(
            theta_test_cpu[:, bin_index], theta_pred[:, bin_index]
        )
        best_model_info["metrics_per_bin"].append((r2_bin, rmse_bin, mae_bin))

    r2 = r2_score(theta_test_cpu, theta_pred)
    rmse = root_mean_squared_error(theta_test_cpu, theta_pred)
    mae = mean_absolute_error(theta_test_cpu, theta_pred)
    best_model_info["metrics"] = (r2, rmse, mae, best_val_loss)

    # pairplots reionisation history
    theta_mean = theta_test.cpu().mean(dim=1)  # mean across bins for each observation
    low_index = theta_mean.argmin()
    mid_index = (theta_mean - theta_mean.mean()).abs().argmin()
    high_index = theta_mean.argmax()

    z_centers = [158.45, 173.45, 188.45]
    r_range = ["151.0-165.9", "166.0-180.9", "181.0-195.9"]

    scenario_samples = {}

    pairplot_figs = {}
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))

    for name, index, color in [
        ("low", low_index, "red"),
        ("mid", mid_index, "green"),
        ("high", high_index, "blue"),
    ]:
        with torch.no_grad():
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

        fig_pair, ax = pairplot(samples, labels=["xHI_z1", "xHI_z2", "xHI_z3"])
        pairplot_figs[name] = fig_pair
        fig_pair.savefig(
            f"models/ensemble-sbi/{data_type}/pairplot_{name}_xHI.png"
        )  # also available via mlflow

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

    # tarp plot redo, since new posterior
    num_tarp_samples = 500
    tarp_index = torch.randperm(len(theta_test))[:num_tarp_samples]

    prior_samples = theta_test[tarp_index]  # ground truth theta
    prior_predictives = x_test[tarp_index]  # corresponding observations

    ecp, alpha = run_tarp(
        prior_samples,
        prior_predictives,
        best_posterior,
        references=None,
        num_posterior_samples=1000,
        use_batched_sampling=True,
    )

    atc, ks_pval = check_tarp(ecp.cpu(), alpha.cpu())
    tarp_fig, _ = plot_tarp(ecp.cpu(), alpha.cpu())
    tarp_fig.savefig(f"models/ensemble-sbi/{data_type}/tarp_plot.png")
    tracker.add_figure("tarp_plot", tarp_fig)

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
                "tarp_atc": float(tarp_dict_atc.get(best_num, float("nan"))),
                "tarp_ks_pval": float(tarp_dict.get(best_num, float("nan"))),
                "val_loss": float(best_model_info["metrics"][3]),
            },
            f,
            indent=4,
        )

    mlflow.log_metrics(
        {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "tarp_atc": float(tarp_dict_atc.get(best_num, float("nan"))),
            "tarp_ks_pval": float(tarp_dict.get(best_num, float("nan"))),
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

mlflow.end_run()
