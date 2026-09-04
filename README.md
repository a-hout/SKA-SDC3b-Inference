# SKA SDC3b Inference
Check the report and appendices and the presentation for the full explanation.
## Technologies
The models are implemented in Python 3.12.

[uv package manager](https://docs.astral.sh/uv/), to facilitate and speed up package management.

Python packages:
* [MLflow](https://mlflow.org/classical-ml), experiment tracking, metric/artifact logging, and training visualization; runs on port 8481
* [sbi](https://sbi.readthedocs.io/en/latest/), simulation-based inference (NPE, EnsemblePosterior, TARP diagnostics)
* [PyTorch](https://docs.pytorch.org/docs/stable/index.html) / torchvision, deep learning and the base ResNet-18 architecture
* numpy
* scikit-learn, cross-validation (KFold), nearest-neighbour lookup for PPC, and R²/RMSE/MAE metrics
* scipy, kernel density estimation for P(x|z) plots
* tqdm, progress bars for posterior sampling loops
* Matplotlib, all evaluation plots (TARP curves, pairplots, reionization history, PPC scatter plots)
* Jupyter notebooks and Polars, exploratory data analysis

## Data
Two summary statistics are used, both derived from 21cmFAST mock observations (following Cerardi et al., same dataset source as Michele Bianco's simulations):

* PS2D,cylindrically-averaged 2D power spectra, shape (3, 10, 10) per observation (3 redshift bins x 10x10 (k perpendicular, k parallel) grid), 20 independent noise realizations per parameter set.
* Betti numbers, structural summary (b0, b1, b2) across 21 density thresholds, shape (3, 3, 21) per observation, same 20 noise realizations per parameter set.

Both share the same target parameter theta: the neutral hydrogen fraction xHI across the same three redshift bins.

Six architectures are trained in total: SBI and a 5-ensemble model SBI, for each of PS2D, Betti, and a joint PS2D+Betti configuration.

## Architecture
The PS2D embedding is a modified ResNet-18 (reduced first-layer kernel/stride/padding, max-pool replaced with identity, given the small 10x10 input), pretrained separately via MSE regression against theta, then frozen and used as a fixed feature extractor for the flow. Activation: ReLU.

The Betti embedding is a small MLP (flatten → linear → LayerNorm → LeakyReLU → dropout → linear → LeakyReLU → linear), trained jointly with the normalizing flow rather than pretrained separately.

A joint embedding. both of the above run in parallel on their respective slice of the flattened, concatenated observation, then their outputs are concatenated and passed through a fusion MLP (same architecture pattern as the Betti embedding).

The normalizing flow (NPE) is trained by minimizing the negative log-likelihood of the true theta under the estimated posterior.

## Testing
Each candidate model is evaluated with two complementary calibration checks, plus standard point-estimate metrics:

* TARP (parameter-space calibration) this will check whether the posterior's credible regions have the correct coverage across the test set. Computed over 20 realizations per model; mean p-value is the primary criterion for selecting the best of 30 independently trained trials (and the best 5 for ensembling).
* PPC (observation-space calibration). Since the 21cmFAST simulator isn't available at inference time, a k-nearest-neighbour lookup in parameter space substitutes for it. Reported via mean |z|-score, mean fractional residual, and mean nearest-neighbour distance.
* R2, RMSE, MAE are computed on the posterior mean as a point estimate, both overall and per redshift bin, against a held-out global test set never used during training or hyperparameter search.

A well-performing model is expected to converge to a low, stable NPE training loss, a TARP p-value comfortably above the standard $\alpha$=0.05 threshold, and low PPC residuals across all three evaluation scenarios (fast/average/slow reionization).

## Pipeline
Each data type has two entry points:

```bash
python src/sbi/sbi_pipeline.py <data_type> <best_params_path>
python src/sbi/sbi_ensemble_pipeline.py <data_type> <best_params_path>
```

where `<data_type>` is one of `ps2d`, `betti`, or `ps2d+betti`.

Stages, in order:
1. Init will load the dataset, split into train/validation and a held-out global test set.
2. Gridsearch (single pipeline only, skipped if `best_params_path` is given) with k-fold cross-validation over a hyperparameter grid, selecting by mean validation loss.
3. 30-trial run where we train 30 independently seeded models with the chosen hyperparameters; select the best by mean TARP p-value (single pipeline), or the best 5 to combine into an `EnsemblePosterior` (ensemble pipeline).
4. Evaluation -> R2/RMSE/MAE, TARP, and PPC on the global test set; all metrics and plots logged as MLflow artifacts.

Hyperparameters found by gridsearch are saved to `models/sbi/<data_type>/best_params.json` and can be reused directly via the second CLI argument.
