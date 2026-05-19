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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available() else "cpu"  # when running locally
_ = torch.manual_seed(26)

dataset = ps2d_loader.PS2DDataset("datasets/pk_128_train_250325_noise/memmap/")
theta, x = dataset[0:4000]  # test

low = torch.tensor([0.0, 0.0, 0.0])
high = torch.tensor(
    [1.0, 1.0, 1.0]
)  # eda showed  0.974121 ┆ 0.956543 ┆ 0.94043 but that was from a sample so we put 1, 1, 1 to be safe
prior = BoxUniform(low=low, high=high)

embedding = ps2d_compressor.PS2DCompressor()  # custom embed, can specify output shape

neural_posterior = posterior_nn(
    model="maf", embedding_net=embedding, z_score_x=None
)  # maf, mdn, etc. -> hyperparameter
trainer = NPE(density_estimator=neural_posterior, prior=prior)

density_estimator = trainer.append_simulations(theta, x).train(
    training_batch_size=128,
    show_train_summary=True,
)
posterior = trainer.build_posterior(density_estimator)

# pairplot
# tarp and sbc
