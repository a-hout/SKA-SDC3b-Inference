"""
In the benchmarks folder we normally have a bunch of benchmarks for training/testing a pipeline and also for inferrence.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="ticks", context="notebook")

def inferrence_plotting(cpu_json, cuda_json):
    """
    Plot a duo table for CPU and CUDA each with four plots
    Left plot is relative times for batch sizes for 1, 16, 32, 64, 128, 256. Not plotting the raw numbers but rather the speed increase compared to 1-sample sampling
    Right plot relative times for percentage of the dataset as batch size. Same principle as left plot.

    Each plot will plot the three data types, top plots the singular and bottom plots the ensemble posterior sampling
    """

    batches = list(cpu_json[0]["ps2d"]["singular"].keys()) #same for all plots

    #cpu
    cpu_data = cpu_json[0]
    for data_type in ["ps2d", "betti", "ps2d+betti"]:
        for pipeline in ["singular", "ensemble"]:
                mean_values = np.mean([np.array(list(cpu_json[i][data_type][pipeline].values())) for i in range(len(cpu_json))], axis=0)
                cpu_data[data_type][pipeline] = dict(zip(batches, mean_values))

    cpu_figure, cpu_axs = plt.subplots(2, 2, figsize=(12,7), dpi=150)

    cpu_x_singular_left, cpu_x_singular_right = batches[:5], [f"{int((float(batch)/1000)*100)}%" for batch in batches[6:]]
    cpu_y_singular = [np.array(list(cpu_data[data_type]["singular"].values()))[0]/np.array(list(cpu_data[data_type]["singular"].values())) for data_type in ["ps2d", "betti", "ps2d+betti"]]
    cpu_y_singular_left, cpu_y_singular_right = [y[:5] for y in cpu_y_singular], [y[6:] for y in cpu_y_singular]
    cpu_y_ensemble = [np.array(list(cpu_data[data_type]["ensemble"].values()))[0]/np.array(list(cpu_data[data_type]["ensemble"].values())) for data_type in ["ps2d", "betti", "ps2d+betti"]]
    cpu_y_ensemble_left, cpu_y_ensemble_right = [y[:5] for y in cpu_y_ensemble], [y[6:] for y in cpu_y_ensemble]

    cpu_axs[0,0].set_ylabel('Singular posterior')
    cpu_axs[0, 0].plot(cpu_x_singular_left, cpu_y_singular_left[0], label='PS2D')
    cpu_axs[0, 0].plot(cpu_x_singular_left, cpu_y_singular_left[1], label='Betti')
    cpu_axs[0, 0].plot(cpu_x_singular_left, cpu_y_singular_left[2], label='PS2D + Betti')
    cpu_axs[0,0].grid(axis='x')
    cpu_axs[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cpu_axs[0, 0].set_ylim(bottom=1)

    cpu_axs[0, 1].plot(cpu_x_singular_right, cpu_y_singular_right[0])
    cpu_axs[0, 1].plot(cpu_x_singular_right, cpu_y_singular_right[1])
    cpu_axs[0, 1].plot(cpu_x_singular_right, cpu_y_singular_right[2])
    cpu_axs[0,1].grid(axis='x')
    cpu_axs[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cpu_axs[0, 1].set_ylim(bottom=1)

    cpu_axs[1,0].set_ylabel('5-Ensemble posterior')
    cpu_axs[1, 0].set_xlabel("Absolute batch sizes")
    cpu_axs[1, 0].plot(cpu_x_singular_left, cpu_y_ensemble_left[0])
    cpu_axs[1, 0].plot(cpu_x_singular_left, cpu_y_ensemble_left[1])
    cpu_axs[1, 0].plot(cpu_x_singular_left, cpu_y_ensemble_left[2])
    cpu_axs[1,0].grid(axis='x')
    cpu_axs[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cpu_axs[1, 0].set_ylim(bottom=1)

    cpu_axs[1, 1].set_xlabel("Proportional batch sizes (in %)")
    cpu_axs[1, 1].plot(cpu_x_singular_right, cpu_y_ensemble_right[0])
    cpu_axs[1, 1].plot(cpu_x_singular_right, cpu_y_ensemble_right[1])
    cpu_axs[1, 1].plot(cpu_x_singular_right, cpu_y_ensemble_right[2])
    cpu_axs[1,1].grid(axis='x')
    cpu_axs[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cpu_axs[1, 1].set_ylim(bottom=1)

    cpu_figure.suptitle("CPU: Batched sampling of 1000 observations per data type")
    cpu_figure.legend(loc="outside right upper", frameon=False)
    #cpu_figure.show()

    #cuda
    cuda_data = cuda_json[0]
    for data_type in ["ps2d", "betti", "ps2d+betti"]:
        for pipeline in ["singular", "ensemble"]:
                mean_values = np.mean([np.array(list(cuda_json[i][data_type][pipeline].values())) for i in range(len(cuda_json))], axis=0)
                cuda_data[data_type][pipeline] = dict(zip(batches, mean_values))

    cuda_figure, cuda_axs = plt.subplots(2, 2, figsize=(12,7), dpi=150)
    cuda_x_singular_left, cuda_x_singular_right = batches[:5], [f"{int((float(batch)/1000)*100)}%" for batch in batches[6:]]
    cuda_y_singular = [np.array(list(cuda_data[data_type]["singular"].values()))[0]/np.array(list(cuda_data[data_type]["singular"].values())) for data_type in ["ps2d", "betti", "ps2d+betti"]]
    cuda_y_singular_left, cuda_y_singular_right = [y[:5] for y in cuda_y_singular], [y[6:] for y in cuda_y_singular]
    cuda_y_ensemble = [np.array(list(cuda_data[data_type]["ensemble"].values()))[0]/np.array(list(cuda_data[data_type]["ensemble"].values())) for data_type in ["ps2d", "betti", "ps2d+betti"]]
    cuda_y_ensemble_left, cuda_y_ensemble_right = [y[:5] for y in cuda_y_ensemble], [y[6:] for y in cuda_y_ensemble]


    cuda_axs[0,0].set_ylabel('Singular posterior')
    cuda_axs[0, 0].plot(cuda_x_singular_left, cuda_y_singular_left[0], label='PS2D')
    cuda_axs[0, 0].plot(cuda_x_singular_left, cuda_y_singular_left[1], label='Betti')
    cuda_axs[0, 0].plot(cuda_x_singular_left, cuda_y_singular_left[2], label='PS2D + Betti')
    cuda_axs[0, 0].grid(axis='x')
    cuda_axs[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cuda_axs[0, 0].set_ylim(bottom=1)

    cuda_axs[0, 1].plot(cuda_x_singular_right, cuda_y_singular_right[0])
    cuda_axs[0, 1].plot(cuda_x_singular_right, cuda_y_singular_right[1])
    cuda_axs[0, 1].plot(cuda_x_singular_right, cuda_y_singular_right[2])
    cuda_axs[0, 1].grid(axis='x')
    cuda_axs[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cuda_axs[0, 1].set_ylim(bottom=1)

    cuda_axs[1,0].set_ylabel('5-Ensemble posterior')
    cuda_axs[1, 0].set_xlabel("Absolute batch sizes")
    cuda_axs[1, 0].plot(cuda_x_singular_left, cuda_y_ensemble_left[0])
    cuda_axs[1, 0].plot(cuda_x_singular_left, cuda_y_ensemble_left[1])
    cuda_axs[1, 0].plot(cuda_x_singular_left, cuda_y_ensemble_left[2])
    cuda_axs[1, 0].grid(axis='x')
    cuda_axs[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cuda_axs[1, 0].set_ylim(bottom=1)

    cuda_axs[1, 1].set_xlabel("Proportional batch sizes (in %)")
    cuda_axs[1, 1].plot(cuda_x_singular_right, cuda_y_ensemble_right[0])
    cuda_axs[1, 1].plot(cuda_x_singular_right, cuda_y_ensemble_right[1])
    cuda_axs[1, 1].plot(cuda_x_singular_right, cuda_y_ensemble_right[2])
    cuda_axs[1, 1].grid(axis='x')
    cuda_axs[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    cuda_axs[1, 1].set_ylim(bottom=1)

    cuda_figure.suptitle("CUDA: Batched sampling of 1000 observations per data type")
    cuda_figure.legend(loc="outside right upper", frameon=False)

    #plotting
    #plt.show()
    cpu_figure.savefig('plots/cpu_inference_benchmark.png', dpi=300)
    cuda_figure.savefig('plots/cuda_inference_benchmark.png', dpi=300)

def inferrence_analysis(cpu_json, cuda_json):
    pass

if __name__ == "__main__":
    cpu = ["13-8-57", "13-9-14", "13-9-31", "13-9-49", "13-10-6"]
    cpu_json = []
    cuda = ["13-8-55", "13-9-12", "13-9-29", "13-9-46", "13-10-2"]
    cuda_json = []
    for cpu_file in cpu:
        with open(f"benchmarks/cpu-inferring-1000-obs-benchmark-{cpu_file}.json") as f:
            cpu_json.append(json.load(f))
        f.close()

    for cuda_file in cuda:
        with open(f"benchmarks/cuda-inferring-1000-obs-benchmark-{cuda_file}.json") as f:
            cuda_json.append(json.load(f))
        f.close()

    inferrence_plotting(cpu_json, cuda_json)
    inferrence_analysis(cpu_json, cuda_json)
