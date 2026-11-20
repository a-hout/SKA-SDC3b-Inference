"""
In the benchmarks folder we normally have a bunch of benchmarks for training/testing a pipeline and also for inferrence.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sns.set_theme(style="ticks", context="notebook")

def inferrence_plotting(json, device):
    unit, scale= "ms", 1e3

    runs = json
    data_type, pipeline_type = list(runs[0]), list(runs[0][next(iter(runs[0]))])
    data = {m: {k: np.array([r[m][k] for r in runs]) * scale for k in pipeline_type} for m in data_type}
    n = len(runs)

    #plot
    rng = np.random.default_rng(0)

    def draw(ax, positions, values, color, width):
        """box and per-run dots with mean and std marker."""
        bp = ax.boxplot(values, positions=positions, widths=width, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.35, edgecolor=color)
        for pos, v in zip(positions, values):
            ax.scatter(pos + rng.uniform(-width / 3, width / 3, len(v)), v, s=12, color=color, alpha=0.7, zorder=3)
            ax.errorbar(pos, v.mean(), yerr=v.std(ddof=1), fmt="D", color="black", ms=5, capsize=4, zorder=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(data_type))
    colors = {"singular": "#3b6fb6", "ensemble": "#d9822b"}

    for j, k in enumerate(pipeline_type):
        draw(ax1, x + (j - 0.5) * 0.35, [data[m][k] for m in data_type], colors[k], 0.28)
        ax1.scatter([], [], color=colors[k], label=k)
    ax1.errorbar([], [], yerr=[], fmt="D", color="black", capsize=4, label="mean ± std")
    ax1.set(ylabel=f"Time per observation [{unit}]",
            title=f"Distribution of 1-obs inferrence time ({n} runs, mean over 10,000 samples)")
    ax1.legend(frameon=False)

    #ratio per run between singular and ensemble counterpart
    draw(ax2, x, [data[m]["ensemble"] / data[m]["singular"] for m in data_type], "#666666", 0.5)
    ax2.axhline(1, color="red", ls="--", lw=1)
    ax2.set(ylabel="ensemble / singular time", title="Ensemble overhead compared to singular")

    for ax in (ax1, ax2):
        ax.set_xticks(x, data_type)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{device}-inferrence.png", dpi=300)
    inferrence_analysis(data, n, unit, device)

def inferrence_analysis(data, n, unit, device):
    lines = [f"Inference time per observation [{unit}] - {device}",
                f"{n} runs, over 10,000 samples", ""]

    for m in data:
        for k in data[m]:
            x = data[m][k]
            lo, hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
            lines.append(f"{m} | {k}: mean={x.mean():.3f}  std={x.std(ddof=1):.3f}  "
                            f"median={np.median(x):.3f}  min={x.min():.3f}  max={x.max():.3f}  "
                            f"95% CI of mean=[{lo:.3f}, {hi:.3f}]")
        r = data[m]["ensemble"] / data[m]["singular"]
        lines.append(f"{m} | ensemble/singular: mean={r.mean():.2f}  std={r.std(ddof=1):.2f}")
        lines.append("")

    with open(f"plots/{device}-inferrence.txt", "w") as f:
        f.write("\n".join(lines))

def compare_devices(runs_cpu, runs_cuda, scale=1e3, unit="ms"):
    data_type, pipeline_type = list(runs_cpu[0]), list(runs_cpu[0][next(iter(runs_cpu[0]))])
    get = lambda runs, m, k: np.array([r[m][k] for r in runs]) * scale

    lines = [f"CPU vs CUDA, inference time per observation [{unit}]",
             "speedup = CPU / CUDA (>1: CUDA faster, <1: CUDA slower)", ""]
    speedup, err = {}, {}
    for k in pipeline_type:
        speedup[k], err[k] = [], []
        for m in data_type:
            c, g = get(runs_cpu, m, k), get(runs_cuda, m, k)
            s = c.mean() / g.mean()
            e = s * np.sqrt((stats.sem(c) / c.mean())**2 + (stats.sem(g) / g.mean())**2)  #standard errror of the ratio
            speedup[k].append(s); err[k].append(e)
            lines.append(f"{m} | {k}: CPU={c.mean():.3f}  CUDA={g.mean():.3f}  speedup={s:.2f} ± {e:.2f}")
    open("plots/cpu-vs-cuda.txt", "w").write("\n".join(lines))

    colors = {"singular": "#3b6fb6", "ensemble": "#d9822b"}
    x = np.arange(len(data_type))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, k in enumerate(pipeline_type):
        ax.bar(x + (j - 0.5) * 0.35, speedup[k], 0.35, yerr=err[k], capsize=4, color=colors[k], alpha=0.8, label=k)
    ax.axhline(1, color="red", ls="--", lw=1)
    ax.set_xticks(x, data_type)
    ax.set(ylabel="speedup (CPU time / CUDA time)", title="CUDA speedup over CPU")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/cpu-vs-cuda.png", dpi=300)

if __name__ == "__main__":
    cpu_json = []
    cuda_json = []
    for cpu_file in range(30):
        with open(f"benchmarks/inferring/cpu/{cpu_file}.json") as f:
            cpu_json.append(json.load(f))
        f.close()
    inferrence_plotting(cpu_json, "cpu")
    for cuda_file in range(30):
        with open(f"benchmarks/inferring/cuda/{cuda_file}.json") as f:
            cuda_json.append(json.load(f))
        f.close()
    inferrence_plotting(cuda_json, "cuda")
    compare_devices(cpu_json, cuda_json)
