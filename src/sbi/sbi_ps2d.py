"""
SBI implementation for the PS2D dataset pk_128_train_250325_noise
"""

# import mlflow
import torch
from compressor import ps2d_compressor

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available() else "cpu"  # when running locally
_ = torch.manual_seed(26)

# to speed up training, use f8, not even f16. if too weird, try with f16

dataloader = torch.lo

embedding = ps2d_compressor.PS2DCompressor()
