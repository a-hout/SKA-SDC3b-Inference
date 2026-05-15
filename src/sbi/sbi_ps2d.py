"""
SBI implementation for the PS2D dataset pk_128_train_250325_noise
"""

import mlflow

PS2D_PATH = "datasets/pk_128_train_250325_noise/"  # always run from root of workspace, not from src/

mlflow.set_experiment("SBI PS2D")
mlflow.pytorch.autolog()

with mlflow.start_run():
    pass
