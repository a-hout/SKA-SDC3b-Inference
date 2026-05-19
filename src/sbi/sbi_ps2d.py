"""
SBI implementation for the PS2D dataset pk_128_train_250325_noise
"""

# import mlflow
import torch
from compressor import ps2d_compressor
from dataloader import ps2d_loader
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available() else "cpu"  # when running locally
_ = torch.manual_seed(26)

dataset = ps2d_loader.PS2DDataset("datasets/pk_128_train_250325_noise/memmap/")
theta, x = dataset[:]

low = torch.tensor([0.0, 0.0, 0.0])
high = torch.tensor(
    [1.0, 1.0, 1.0]
)  # eda showed  0.974121 ┆ 0.956543 ┆ 0.94043 but that was from a sample so we put 1, 1, 1 to be safe
prior = BoxUniform(low=low, high=high)

r2_scores = []
rmse_scores = []
mae_scores = []
# we will be using cross validation with k = 3 to train the posterior
cv = KFold(n_splits=3, shuffle=True, random_state=26)
for fold, (train_index, test_index) in enumerate(cv.split(range(len(dataset)))):
    print(f"fold {fold + 1}")

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
        embedding_net=ps2d_compressor.PS2DCompressor(),  # custom embed, can specify output shape
        z_score_x=None,
    )  # maf, mdn, etc. -> hyperparameter
    trainer = NPE(density_estimator=neural_posterior, prior=prior)

    density_estimator = trainer.append_simulations(theta_train, x_train).train(
        training_batch_size=128,
        show_train_summary=True,
    )
    posterior = trainer.build_posterior(density_estimator)

    theta_pred = []
    for x_obs in x_test:
        samples = posterior.sample(
            (1000,), x=x_obs
        )  # 1000 samples from the approximate distribution given an observation

        theta_pred.append(
            samples.mean(dim=0)
        )  # average (mean here) of the 1000 samples appended to prediction list

    theta_pred = torch.stack(theta_pred)

    r2_scores.append(r2_score(theta_test, theta_pred))
    rmse_scores.append(root_mean_squared_error(theta_test, theta_pred))
    mae_scores.append(mean_absolute_error(theta_test, theta_pred))


print("R2 and RMSE scores of cross validators...")
for index, (r2, rmse, mae) in enumerate(zip(r2_scores, rmse_scores, mae_scores)):
    print(f"\tcv={index}:\tR2={r2}\tRMSE={rmse}\tMAE={mae}")


# pairplot
# tarp and sbc
