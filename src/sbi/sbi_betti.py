"""
SBI implementation for the Betti numbers dataset
"""

import json
from itertools import product

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from compressor import betti_compressor
from dataloader import betti_loader
from mlflowadapter import MLflowAdapter
from sbi.analysis import pairplot
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics import check_tarp, run_tarp
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from tqdm import tqdm

mlflow.set_experiment("sbi_betti")
mlflow.end_run()

device = (
    "cuda:2"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
_ = torch.manual_seed(26)

dataset = betti_loader.BettiDataset("datasets/betti/", device=device)
theta, _ = dataset[:]

low = torch.tensor([0.0, 0.0, 0.0], device=device)  # stays the same as for ps2d
high = (
    theta[0, :].max(dim=0).values.detach().clone().to(device)
)  # convert to shape (3,)
prior = BoxUniform(low=low, high=high, device=device)

cv_data = {
    "sbi_validation_scores": [],
    "metrics": (),
    "metrics_per_bin": [],
    "split_index": (),
}

global_best_mean_val_loss = float("inf")

# we will be using cross validation with k = 3 to train the posterior
for (
    summary_dim,
    model,
    num_transforms,
    hidden_features,
) in product(  # 2*2*2*2 = 16 runs -> *3 folds = 48 runs
    [16],  # 4 too small, 8 may be interesting but already 2 many h-params
    [
        "nsf"
    ],  # nsf may be slower and more prone to overfitting but could provide better results
    [8],  # num transforms for control flow of model
    [128],  # how big the hidden layers are (width of flow)
):
    combination_val_losses = []
    combination_best = {}  # this is the best fold for a given gridsearch combination
    with mlflow.start_run(
        run_name=f"{model}_sd{summary_dim}_nt{num_transforms}_hf{hidden_features}"
    ):
        cv = KFold(n_splits=3, shuffle=True, random_state=26)
        for fold, (train_index, test_index) in enumerate(cv.split(range(len(dataset)))):
            with mlflow.start_run(run_name=f"fold_{fold + 1}", nested=True):
                tracker = MLflowAdapter(mlflow)

                mlflow.log_params(
                    {
                        "fold": fold + 1,
                        "summary_dim": summary_dim,
                        "model": model,
                        "num_transforms": num_transforms,
                        "hidden_features": hidden_features,
                    }
                )

                train_set = Subset(dataset, train_index)
                test_set = Subset(dataset, test_index)

                theta_train, x_train = zip(
                    *[train_set[i] for i in range(len(train_set))]
                )
                theta_train = torch.stack(theta_train)
                x_train = torch.stack(x_train)
                theta_test, x_test = zip(*[test_set[i] for i in range(len(test_set))])
                theta_test = torch.stack(theta_test)
                x_test = torch.stack(x_test)

                neural_posterior = posterior_nn(
                    model=model,
                    embedding_net=betti_compressor.BettiCompressor(
                        summary_dim=summary_dim
                    ),
                    z_score_x="independent",  # contrary to ps2d, we have extreme outliers so need to normalize
                    hidden_features=hidden_features,
                    num_transforms=num_transforms,
                )

                trainer = NPE(
                    density_estimator=neural_posterior,
                    prior=prior,
                    device=device,
                    tracker=tracker,  # built in by sbi for the tracker with adapter
                )

                density_estimator = trainer.append_simulations(
                    theta_train, x_train
                ).train(
                    training_batch_size=512,
                    show_train_summary=True,
                )

                posterior = trainer.build_posterior(
                    density_estimator,  # sample_with="mcmc"
                )  # otherwise betti posterior will sample outside prior

                val_loss = trainer._summary["best_validation_loss"][0]
                mlflow.log_metric("val_loss", val_loss)
                combination_val_losses.append(val_loss)

                if not combination_best or val_loss < combination_best["val_loss"]:
                    combination_best = {
                        "val_loss": val_loss,
                        "density_estimator": density_estimator,
                        "posterior": posterior,
                        "split_index": (train_index, test_index),
                    }

        # here we compare combinations by mean val_loss for folds
        mean_val_loss = sum(combination_val_losses) / len(combination_val_losses)
        mlflow.log_metric("mean_val_loss", mean_val_loss)  # log outside fold runs
        mlflow.log_params(
            {
                "summary_dim": summary_dim,
                "model": model,
                "num_transforms": num_transforms,
                "hidden_features": hidden_features,
            }
        )
    if mean_val_loss < global_best_mean_val_loss:
        global_best_mean_val_loss = mean_val_loss
        cv_data["sbi_density_estimator"] = combination_best["density_estimator"]
        cv_data["sbi_posterior"] = combination_best["posterior"]
        cv_data["split_index"] = combination_best["split_index"]
        cv_data["best_val_loss"] = combination_best["val_loss"]
        cv_data["best_params"] = {
            "summary_dim": summary_dim,
            "model": model,
            "num_transforms": num_transforms,
            "hidden_features": hidden_features,
            "mean_val_loss": mean_val_loss,
        }

torch.save(cv_data["sbi_posterior"], "models/sbi/betti/best_posterior.pt")
torch.save(
    cv_data["sbi_density_estimator"].state_dict(),
    "models/sbi/betti/best_density_estimator.pt",
)

#### calibration testing and visuals of best model
with mlflow.start_run(run_name="best_model_eval"):
    tracker = MLflowAdapter(mlflow)
    mlflow.log_params(cv_data["best_params"])

    test_set = Subset(dataset, cv_data["split_index"][1])

    theta_test, x_test = zip(*[test_set[i] for i in range(len(test_set))])
    theta_test = torch.stack(theta_test)
    x_test = torch.stack(x_test)

    posterior = cv_data["sbi_posterior"]
    theta_pred = []
    # print("Sampling test observations...")
    for x_obs in tqdm(x_test, desc="Sampling test observations"):
        with torch.no_grad():
            samples = posterior.sample(
                (500,), x=x_obs, show_progress_bars=False, reject_outside_prior=False
            )  # 500 samples from the approximate distribution given an observation

            theta_pred.append(
                samples.mean(dim=0).cpu()
            )  # average (mean here) of the 1000 samples appended to prediction list
            torch.cuda.empty_cache()

    theta_pred = torch.stack(theta_pred).cpu()  # back to cpu to convert to numpy
    theta_test_cpu = theta_test.cpu()

    cv_data["metrics_per_bin"] = []
    for bin_index in range(3):  # getting metrics per redshift range
        r2_bin = r2_score(theta_test_cpu[:, bin_index], theta_pred[:, bin_index])
        rmse_bin = root_mean_squared_error(
            theta_test_cpu[:, bin_index], theta_pred[:, bin_index]
        )
        mae_bin = mean_absolute_error(
            theta_test_cpu[:, bin_index], theta_pred[:, bin_index]
        )
        cv_data["metrics_per_bin"].append((r2_bin, rmse_bin, mae_bin))

    r2 = r2_score(theta_test_cpu, theta_pred)
    rmse = root_mean_squared_error(theta_test_cpu, theta_pred)
    mae = mean_absolute_error(theta_test_cpu, theta_pred)
    cv_data["metrics"] = (r2, rmse, mae, cv_data["best_val_loss"])

    # pairplot and reionisation history
    theta_mean = theta_test.cpu().mean(dim=1)  # mean across bins for each observation
    low_index = theta_mean.argmin()
    mid_index = (theta_mean - theta_mean.mean()).abs().argmin()
    high_index = theta_mean.argmax()

    z_centers = [158.45, 173.45, 188.45]
    r_range = ["151.0-165.9", "166.0-180.9", "181.0-195.9"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairplot_figs = {}
    for name, index, color in [
        ("low", low_index, "red"),
        ("mid", mid_index, "green"),
        ("high", high_index, "blue"),
    ]:
        print(name)
        with torch.no_grad():
            samples = posterior.sample(
                (2500,),
                x=x_test[index],
                show_progress_bars=False,
                reject_outside_prior=False,
            ).cpu()
        torch.cuda.empty_cache()

        x_z = samples.mean(dim=0).numpy()  # mean of distribution
        lower = samples.quantile(0.16, dim=0).numpy()  # -1sigma from distribution
        upper = samples.quantile(0.84, dim=0).numpy()  # +1sigma
        print(f"{name}: true theta = {theta_test[index].cpu().numpy()}")

        for i, zbin in enumerate(r_range):
            mlflow.log_metrics(
                {
                    f"xHI_mean_{name}_z{i + 1}": float(x_z[i]),
                    f"xHI_lower_{name}_z{i + 1}": float(lower[i]),
                    f"xHI_upper_{name}_z{i + 1}": float(upper[i]),
                }
            )

        # plto reionisation fraction history
        axes[0].plot(z_centers, x_z, "o-", color=color, label=f"{name} xHI")
        axes[0].fill_between(z_centers, lower, upper, color=color, alpha=0.2)

        # uncertainty P(x|z) for each redshift range
        for i in range(3):
            kde = gaussian_kde(
                samples[:, i].numpy()
            )  # shape of sample is (sample number, 3) for each z of xHI

            x_range = np.linspace(0, 1, 200)
            axes[1].plot(
                x_range,
                kde(x_range),
                color=color,
                linestyle=["-", "--", ":"][i],
                label=f"{name} {r_range[i]}",
            )

        fig_pair, ax = pairplot(samples, labels=["xHI_z1", "xHI_z2", "xHI_z3"])
        pairplot_figs[name] = fig_pair
        fig_pair.savefig(
            f"models/sbi/betti/pairplot_{name}_xHI.png"
        )  # also available via mlflow

    for boundary in [165.9, 180.9]:
        axes[0].axvline(boundary, color="gray", linestyle="--", alpha=0.5)

    axes[0].set_xlabel("Frequency (MHz)")
    axes[0].set_ylabel("mean_xHI")
    axes[0].set_title("Reionization History x(z)")
    axes[0].legend()

    axes[1].set_xlabel("mean_xHI")
    axes[1].set_ylabel("P(x|z)")
    axes[1].set_title("Posterior per redshift range")
    axes[1].legend(fontsize=7)

    fig.savefig("models/sbi/betti/reionization_history.png")
    tracker.add_figure("reionization_history", fig)

    # tarp, impossible at the moment
    """
    num_tarp_samples = 200
    tarp_index = torch.randperm(len(theta_test))[:num_tarp_samples]

    prior_samples = theta_test[tarp_index]  # ground truth theta
    prior_predictives = x_test[tarp_index]  # corresponding observations

    with torch.no_grad():
        ecp, alpha = run_tarp(
            prior_samples,
            prior_predictives,
            posterior,
            references=None,
            num_posterior_samples=1000,
            use_batched_sampling=True,
        )

        ecp_cpu = ecp.cpu()
        alpha_cpu = alpha.cpu()

    torch.cuda.empty_cache()

    atc, ks_pval = check_tarp(ecp_cpu, alpha_cpu)
    """

    with open("models/sbi/betti/evaluation.json", "w") as f:
        json.dump(
            {
                "r2": float(cv_data["metrics"][0]),
                "rmse": float(cv_data["metrics"][1]),
                "mae": float(cv_data["metrics"][2]),
                "bin_metrics": {
                    r_range[i]: {
                        "r2": float(cv_data["metrics_per_bin"][i][0]),
                        "rmse": float(cv_data["metrics_per_bin"][i][1]),
                        "mae": float(cv_data["metrics_per_bin"][i][2]),
                    }
                    for i in range(3)
                },
                "val_loss": float(cv_data["metrics"][3]),
                # "tarp_atc": float(atc),
                # "tarp_ks_pval": float(ks_pval),
                "best_params": cv_data["best_params"],
            },
            f,
            indent=4,
        )

    # tarp_fig, ax = plot_tarp(ecp.cpu(), alpha.cpu())

    # tarp_fig.savefig("models/sbi/betti/tarp_plot.png")  # also available via mlflow

    mlflow.log_metrics(
        {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            # "tarp_atc": float(atc),
            # "tarp_ks_pval": float(ks_pval),
        }
    )

    # log per-bin metrics
    for i, zbin in enumerate(r_range):
        mlflow.log_metrics(
            {
                f"r2_{zbin}": float(cv_data["metrics_per_bin"][i][0]),
                f"rmse_{zbin}": float(cv_data["metrics_per_bin"][i][1]),
                f"mae_{zbin}": float(cv_data["metrics_per_bin"][i][2]),
            }
        )

    # log figures

    # tracker.add_figure("tarp_plot", tarp_fig)
    for name in ["low", "mid", "high"]:
        tracker.add_figure(f"pairplot_{name}_xHI", pairplot_figs[name])

mlflow.end_run()
