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

mlflow.set_experiment("sbi_betti")
mlflow.end_run()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
_ = torch.manual_seed(26)

dataset = betti_loader.BettiDataset("datasets/betti/", device=device)
theta, obs = dataset[:]

low = torch.tensor([0.0, 0.0, 0.0], device=device)  # TODO change
high = theta.max(dim=0).values.detach().clone().to(device)  # TODO change
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
    [16, 32],  # 4 too small, 8 may be interesting but already 2 many h-params
    [
        "maf",
        "nsf",
    ],  # nsf may be slower and more prone to overfitting but could provide better results
    [4, 8],  # num transforms for control flow of model
    [64, 128],  # how big the hidden layers are (width of flow)
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
                    z_score_x=None,
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
