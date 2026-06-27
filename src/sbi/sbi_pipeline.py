##################################################################################################################

"""
Pipeline for generic data types (ps2d, betti) training, with the aid of mlflow.

1. Init of parameters and data

2. Gridsearch for best hyperparameters concerning the SBI model.

3. Train again 30 times with best parameters with different seeds to validate stability
train -> get val-loss -> run tarp (with 10 min timeout and 20 realizations) and get calibration results

4. at the end, get the best model with the best calibration (best mean p-val of TARP)

5. Full evaluation with metrics, overal and per z_x
R2
RMSE
MAE
Val loss

pairplots

reionisation history with the three scenarios + the actual mean ground truth for comparison

P(x | z) plot with the three scenarios (3 plots)
"""

##################################################################################################################
# 0. imports
##################
import copy
import json
import os
import signal
import sys
from itertools import product

import matplotlib.pyplot as plt
import mlflow
import numpy as np
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
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics import check_tarp, run_tarp
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
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
        "Normal usage is: python src/sbi/sbi_pipeline.py <data_type> <optional best params path>"
    )
    exit(1)

try:
    with open(sys.argv[2]) as f:  # load path with best model parameters
        BEST_PARAMS = json.load(f)["best_params"]
except (IndexError, FileNotFoundError):
    BEST_PARAMS = None

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
prior = BoxUniform(low=low, high=high, device=device)


def get_compressor(data_type, summary_dim, model=None):
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


##################
# 2. gridsearching
##################
# should be an option to disable this section if already best params found (and then load via json for 30-run)
if BEST_PARAMS is None:
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
        num_bins,
        learning_rate,
    ) in product(  # 2*2*2*2 = 16 runs -> *3 folds = 48 runs
        [3, 16, 32],  # 3 (to narrow to 3 redshift ranges)
        [
            "nsf",
        ],  # nsf may be slower and more prone to overfitting but could provide better results
        [8],  # num transforms for control flow of model
        [128],  # how big the hidden layers are (width of flow)
        [10, 32],
        [5e-4, 1e-4],  # learning rate
    ):
        combination_val_losses = []
        combination_best = {}  # this is the best fold for a given gridsearch combination
        with mlflow.start_run(
            run_name=f"{model}_sd{summary_dim}_nbins{num_bins}_lr{learning_rate}"
        ):
            cv = KFold(n_splits=3, shuffle=True, random_state=26)
            for fold, (train_index, test_index) in enumerate(
                cv.split(range(len(dataset)))
            ):
                with mlflow.start_run(run_name=f"fold_{fold + 1}", nested=True):
                    tracker = MLflowAdapter(mlflow)

                    mlflow.log_params(
                        {
                            "fold": fold + 1,
                            "summary_dim": summary_dim,
                            "model": model,
                            "num_transforms": num_transforms,
                            "hidden_features": hidden_features,
                            "num_bins": num_bins,
                            "learning_rate": learning_rate,
                        }
                    )

                    train_set = Subset(dataset, train_index)
                    test_set = Subset(dataset, test_index)

                    theta_train, x_train = zip(
                        *[train_set[i] for i in range(len(train_set))]
                    )
                    theta_train = torch.stack(theta_train)
                    x_train = torch.stack(x_train)
                    theta_test, x_test = zip(
                        *[test_set[i] for i in range(len(test_set))]
                    )
                    theta_test = torch.stack(theta_test)
                    x_test = torch.stack(x_test)

                    neural_posterior = posterior_nn(
                        model=model,
                        embedding_net=get_compressor(data_type, summary_dim),
                        # z_score_x="none",
                        # z_score_theta="independent",
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

                    density_estimator = trainer.append_simulations(
                        theta_train, x_train
                    ).train(
                        training_batch_size=512,
                        show_train_summary=True,
                        learning_rate=learning_rate,
                    )

                    posterior = trainer.build_posterior(density_estimator)

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
                    "num_bins": num_bins,
                    "learning_rate": learning_rate,
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
                "num_bins": num_bins,
                "learning_rate": learning_rate,
            }
    try:
        print("making data type directory...")
        os.mkdir(f"models/sbi/{data_type}/")
    except OSError:
        print("directory already exists.")
    with open(f"models/sbi/{data_type}/best_params.json", "w") as f:
        json.dump(
            {
                "best_params": cv_data["best_params"],
            },
            f,
            indent=4,
        )
    BEST_PARAMS = cv_data["best_params"]

##################
# 3. 30-run
##################
summary_dim, model, num_transforms, hidden_features, num_bins, learning_rate = (
    BEST_PARAMS.values()
)
global_gen = torch.manual_seed(831)

train_val_dataset, global_test_dataset = random_split(
    (Subset(dataset, range(int(len(dataset) / 1)))),
    [0.8, 0.2],
    generator=global_gen,
)
# for testing, a 15th of the dataset size


model_info = {
    "number": [],
    "val_loss_values": [],
    "tarp_values": [],
    "tarp_atc_values": [],
    "posteriors": [],
    "density_estimators": [],
    "tarp_curves": [],
}

tarp_fig, tarp_axs = plt.subplots(
    6, 5, figsize=(20, 30)
)  # 1 plot per one of the 30 models

with mlflow.start_run(run_name="Best model run"):
    for i in range(1):  # testing with 5, producition with 30
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
            x_train = torch.stack(x_train)

            theta_test, x_test = zip(*[test_set[j] for j in range(len(test_set))])
            theta_test = torch.stack(theta_test)
            x_test = torch.stack(x_test)

            embedding = None

            ### train a resnet model for the trial
            if data_type in ["ps2d", "ps2d+betti"]:
                print("starting training for resnet!!!")
                resnet = ps2d_resnet.PS2DResNet()

                batch_size = 512
                max_epochs = 300

                criterion = (
                    torch.nn.MSELoss()
                )  # mean square error loss for the regression task
                optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-3)

                resnet.to(device)

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
                train_dataset = TensorDataset(x_train_img, theta_train)
                val_dataset = TensorDataset(
                    x_test_img, theta_test
                )  # test is acting as val here in this case
                train_laoder = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )  # shuffling
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )  # no shuffling

                best_val_loss = float("inf")
                best_weights = None
                epochs_without_better_loss = (
                    0  # after 10 epochs with no improvement, we stop
                )

                for epoch in range(max_epochs):
                    print(f"epoch number {epoch}/{max_epochs}")
                    resnet.train()  # train mode on (this means dropout is active) -> dropout should NOT be active after training when evaluating since the normalizing flow wants something it can go back and forth on
                    train_loss = 0.0

                    for x_batch, theta_batch in train_laoder:
                        x_batch, theta_batch = (
                            x_batch.to(device),
                            theta_batch.to(device),
                        )

                        optimizer.zero_grad()  # get rid of old gradients
                        predictions = resnet(x_batch)
                        loss = criterion(
                            predictions, theta_batch
                        )  # get loss of pred vs real
                        loss.backward()  # backward propagation, according to the loss metric
                        optimizer.step()  # this will update weigths, according to the optimizer

                        train_loss += loss.item() * x_batch.size(0)  # batch loss

                    train_loss /= len(train_laoder.dataset)  # total loss of all batches

                    resnet.eval()  # evaluation mode so we can do validation loss
                    val_loss = 0.0
                    with torch.no_grad():
                        for x_batch, theta_batch in val_loader:
                            x_batch, theta_batch = (
                                x_batch.to(device),
                                theta_batch.to(device),
                            )

                            predictions = resnet(x_batch)
                            loss = criterion(predictions, theta_batch)
                            val_loss += loss.item() * x_batch.size(0)

                    val_loss /= len(val_loader.dataset)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = copy.deepcopy(
                            resnet.state_dict()
                        )  # deepcopy since this will change every epoch
                        epochs_without_better_loss = 0
                    else:
                        epochs_without_better_loss += 1

                    if epochs_without_better_loss >= 15:
                        break  # early stopping

                resnet.load_state_dict(best_weights)
                print("val loss: ", best_val_loss)
                resnet.eval()  # set to ecval mode so dropout is off when used by sbi
                for param in resnet.parameters():
                    param.requires_grad = False  # freeze weigts

                print("finished resnet training")
                embedding = get_compressor(data_type, summary_dim, model=resnet)
            if embedding is None:
                embedding = get_compressor(data_type, summary_dim)
            neural_posterior = posterior_nn(
                model=model,
                embedding_net=embedding,
                z_score_x="none" if data_type != "betti" else "independent",
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

            # 20 realisations
            mean_pval_tarp = 0
            successful_tarps = 0
            current_model_curves = []  # for the 20 tarps
            for k in range(20):
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

                    mean_pval_tarp += ks_pval
                    successful_tarps += 1

                    current_model_curves.append(
                        (ecp.cpu().numpy(), alpha.cpu().numpy())
                    )

                except TimeoutError:
                    print(
                        f"could not complte tarp realisation for model {i + 1}, skipping to next one..."
                    )
                    continue  # go to the next run

            if successful_tarps > 0:
                mean_pval_tarp /= successful_tarps
            else:
                mean_pval_tarp = 0.0  # very bad!

            model_info["tarp_values"].append((i + 1, mean_pval_tarp))
            model_info["tarp_curves"].append((i + 1, current_model_curves))

            mlflow.log_metrics({"tarp_pval_mean": float(mean_pval_tarp)})

            model_info["density_estimators"].append((i + 1, density_estimator))
            model_info["posteriors"].append((i + 1, posterior))

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


tarp_dict = dict(model_info["tarp_values"])
tarp_dict_atc = dict(model_info["tarp_atc_values"])

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
best_model_info = {}
with mlflow.start_run(run_name="Best model evaluation"):
    tracker = MLflowAdapter(mlflow)

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
            "x--",
            color=color,
            markersize=10,
            alpha=0.7,
            label=f"{name} true xHI",
        )

        fig_pair, ax = pairplot(samples, labels=["xHI_z1", "xHI_z2", "xHI_z3"])
        pairplot_figs[name] = fig_pair
        fig_pair.savefig(
            f"models/sbi/{data_type}/pairplot_{name}_xHI.png"
        )  # also available via mlflow

    for boundary in [165.9, 180.9]:
        axes.axvline(boundary, color="gray", linestyle="--", alpha=0.5)

    axes.set_xlabel("Frequency (MHz)")
    axes.set_ylabel("mean_xHI")
    axes.set_title("Reionization History x(z)")
    axes.legend()

    fig.savefig(f"models/sbi/{data_type}/reionization_history.png")

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
    fig_pxz.savefig(f"models/sbi/{data_type}/pxz_plot.png")

    # tarp plot redo, since new test data
    def timeout_handler(signum, frame):
        raise TimeoutError("Global TARP timed out")

    global_tarp_curves = []
    mean_global_pval = 0.0
    successful_global_tarps = 0

    for k in range(20):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes

        try:
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

            atc, ks_pval = check_tarp(ecp.cpu(), alpha.cpu())
            signal.alarm(0)

            mean_global_pval += ks_pval
            successful_global_tarps += 1
            global_tarp_curves.append((ecp.cpu().numpy(), alpha.cpu().numpy()))

        except TimeoutError:
            print(f"Global TARP realization {k + 1} timed out, skipping...")
            continue

    if successful_global_tarps > 0:
        mean_global_pval /= successful_global_tarps

    # get final global TARP plot
    tarp_eval_fig, tarp_eval_ax = plt.subplots(figsize=(6, 6))
    tarp_eval_ax.plot(
        [0, 1], [0, 1], color="black", linestyle="--", alpha=0.8, label="Ideal"
    )

    for ecp_vals, alpha_vals in global_tarp_curves:
        tarp_eval_ax.plot(alpha_vals, ecp_vals, color="purple", alpha=0.15, linewidth=1)

    tarp_eval_ax.set_title(f"Global test set TARP: p = {mean_global_pval:.6f})")
    tarp_eval_ax.set_xlabel("Credibility level")
    tarp_eval_ax.set_ylabel("Expected coverage probability")

    tarp_eval_fig.savefig(f"models/sbi/{data_type}/global_tarp_plot.png")
    tracker.add_figure("global_tarp_plot", tarp_eval_fig)

    # save evaluation
    with open(f"models/sbi/{data_type}/evaluation.json", "w") as f:
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

mlflow.end_run()
