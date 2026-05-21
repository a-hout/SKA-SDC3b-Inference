"""
SBI implementation for the PS2D dataset pk_128_train_250325_noise
"""

# import mlflow
import json

import torch
from compressor import ps2d_compressor
from dataloader import ps2d_loader
from sbi.analysis import pairplot
from sbi.analysis.plot import plot_tarp
from sbi.diagnostics import check_tarp, run_tarp
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
_ = torch.manual_seed(26)

dataset = ps2d_loader.PS2DDataset(
    "datasets/pk_128_train_250325_noise/memmap/", device=device
)
theta, x = dataset[:]

low = torch.tensor([0.0, 0.0, 0.0], device=device)
high = theta.max(dim=0).values.detach().clone().to(device)
prior = BoxUniform(low=low, high=high, device=device)


cv_data = {
    "sbi_validation_scores": [],
    "metrics": (),
    "metrics_per_bin": [],
    "split_index": (),
}

# we will be using cross validation with k = 3 to train the posterior
cv = KFold(n_splits=3, shuffle=True, random_state=26)
for fold, (train_index, test_index) in enumerate(cv.split(range(len(dataset)))):
    train_set = Subset(dataset, train_index)
    test_set = Subset(dataset, test_index)

    theta_train, x_train = zip(*[train_set[i] for i in range(len(train_set))])
    theta_train = torch.stack(theta_train)
    x_train = torch.stack(x_train)
    theta_test, x_test = zip(*[test_set[i] for i in range(len(test_set))])
    theta_test = torch.stack(theta_test)
    x_test = torch.stack(x_test)

    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=ps2d_compressor.PS2DCompressor(
            summary_dim=32  # larger vector
        ),  # custom embed, can specify output shape
        z_score_x=None,
    )  # maf, mdn, etc. -> hyperparameter
    trainer = NPE(density_estimator=neural_posterior, prior=prior, device=device)

    density_estimator = trainer.append_simulations(theta_train, x_train).train(
        training_batch_size=512,
        show_train_summary=True,
    )
    posterior = trainer.build_posterior(density_estimator)

    val_loss = trainer._summary["best_validation_loss"][0]
    print(f"Fold {fold + 1} best validation loss: {val_loss}")
    cv_data["sbi_validation_scores"].append(val_loss)

    if fold > 0:
        if val_loss > min(cv_data["sbi_validation_scores"][:-1]):
            continue  # no point continuing this cv if val loss not the lowest

    theta_pred = []
    print("Sampling test observations...")
    for x_obs in x_test:
        samples = posterior.sample(
            (1000,), x=x_obs, show_progress_bars=False
        )  # 1000 samples from the approximate distribution given an observation

        theta_pred.append(
            samples.mean(dim=0)
        )  # average (mean here) of the 1000 samples appended to prediction list

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

    cv_data["sbi_density_estimator"] = density_estimator
    cv_data["sbi_posterior"] = posterior
    cv_data["metrics"] = (r2, rmse, mae, val_loss)
    cv_data["split_index"] = (train_index, test_index)

torch.save(cv_data["sbi_posterior"], "models/sbi/ps2d/best_posterior.pt")
torch.save(
    cv_data["sbi_density_estimator"].state_dict(),
    "models/sbi/ps2d/best_density_estimator.pt",
)

#### calibration testing and visuals
train_set = Subset(dataset, cv_data["split_index"][0])
test_set = Subset(dataset, cv_data["split_index"][1])

theta_train, x_train = zip(*[train_set[i] for i in range(len(train_set))])
theta_train = torch.stack(theta_train)
x_train = torch.stack(x_train)
theta_test, x_test = zip(*[test_set[i] for i in range(len(test_set))])
theta_test = torch.stack(theta_test)
x_test = torch.stack(x_test)

posterior = cv_data["sbi_posterior"]

# pairplot
x_obs = x_test[0]
theta_mean = theta_test.cpu().mean(dim=1)  # mean across bins for each observation
low_index = theta_mean.argmin()
mid_index = (theta_mean - theta_mean.mean()).abs().argmin()
high_index = theta_mean.argmax()
for name, index in [("low", low_index), ("mid", mid_index), ("high", high_index)]:
    samples = posterior.sample((5000,), x=x_test[index], show_progress_bars=False).cpu()
    fig, ax = pairplot(samples, labels=["xHI_z1", "xHI_z2", "xHI_z3"])
    fig.savefig(f"models/sbi/ps2d/pairplot_{name}_xHI.png")


# tarp and sbc
num_tarp_samples = 200
index = torch.randperm(len(theta_test))[:num_tarp_samples]

prior_samples = theta_test[index]  # ground truth theta
prior_predictives = x_test[index]  # corresponding observations

ecp, alpha = run_tarp(
    prior_samples,
    prior_predictives,
    posterior,
    references=None,
    num_posterior_samples=1000,
    use_batched_sampling=True,
)

atc, ks_pval = check_tarp(ecp.cpu(), alpha.cpu())
print(atc, "Should be close to 0")
print(ks_pval, "Should be larger than 0.05")

print("Overal R2, RMSE and MAE scores of best cross validator...")
print(f"\tR2 = {cv_data['metrics'][0]}")
print(f"\tRMSE = {cv_data['metrics'][1]}")
print(f"\tMAE = {cv_data['metrics'][2]}")
print(f"\tVal loss = {cv_data['metrics'][3]}")

r_range = ["151.0-165.9", "166.0-180.9", "181.0-195.9"]
for bin_index in range(3):
    r2_bin, rmse_bin, mae_bin = cv_data["metrics_per_bin"][bin_index]
    print(f"\n\tBin {r_range[bin_index]}")
    print(f"\tR2 = {r2_bin}")
    print(f"\tRMSE = {rmse_bin}")
    print(f"\tMAE = {mae_bin}")


with open("models/sbi/ps2d/cv_metrics.json", "w") as f:
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
            "tarp_atc": float(atc),
            "tarp_ks_pval": float(ks_pval),
        },
        f,
        indent=4,
    )

fig, ax = plot_tarp(ecp.cpu(), alpha.cpu())

fig.savefig("models/sbi/ps2d/tarp_plot.png")
