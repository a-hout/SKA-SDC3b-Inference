# SKA SDC3b Inference
To do every time logging into the machine:

1. source .venv/bin/activate
2. export MLFLOW_SERVER_ALLOWED_HOSTS="157.26.103.124:8481"
3. mlflow server --host 0.0.0.0 --port 8481


## Technologies
The models are implemented in Python 3.12.12.

[uv package manager](https://docs.astral.sh/uv/), to facilitate and speed up package management

Python packages,
* [MLflow](https://mlflow.org/classical-ml), utility for model training, running on port 8481
* [SBI](https://sbi.readthedocs.io/en/latest/), simulation-based inference model
* [PyTorch](https://docs.pytorch.org/docs/stable/index.html), deep learning library
* [Kymatio](https://www.kymat.io/), wavelet scattering transformation
* Polars, data treatment API
* Matplotlib and Seaborn, plotting libraries
* Jupyter notebooks, for exploratory data analysis

## Training
The PS1 and PS2 sets are made using different algorithms, but they share their input characteristics. Our goal is to train two models using one shared architecture, for both the PS1 and PS2 sets.

Loss function:
Activation function:


## Testing
Algorithms are provided by the SKAO in order to gauge the precision of a model relative to the dataset it’s trained on.

We will thus seek to have a low and converging loss function (for both validation and training) and a high score on the testing algorithms. (*TODO* make sure this is correct)

## Pipeline
The pipeline will input the data, train the two models, …

How to set up the pipeline? look at state of the art reproducible models (when available)
