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
