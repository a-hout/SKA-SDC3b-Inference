"""
SBI implementation for the PS2D dataset pk_128_train_250325_noise
"""

# import mlflow
import torch
from compressor import ps2d_compressor

device = "cuda" if torch.cuda.is_available() else "cpu"
_ = torch.manual_seed(26)

PS2D_PATH = "datasets/pk_128_train_250325_noise/"  # always run from root of workspace, not from src/

# to speed up training, use f8, not even f16. if too weird, try with f16

embedding = ps2d_compressor.PS2DCompressor()
