import numpy as np
import sys
import matplotlib.pyplot as plt
from data_processing import gen_result_df
from metric_calc import objective_function

plt.rcParams["pdf.fonttype"] = 42

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"
save_path = str(Path(__file__).parents[1] / "output_plots") + "/"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

ax = plt.figure(figsize=(4, 2.8)).add_subplot(111)
colors = ["#1f77b4", "#d62728"]


def plot_metric(forecasters, result_df):
    args = dict()
    args["x_label"] = "RUM"
    args["y_label"] = "Fraction of Apps"
    args["file_name"] = save_path + "sec_4_data_rep.pdf"
    args["log"] = True

    for forecast_num, forecaster in enumerate(forecasters):
        result_df["Metric"] = result_df.apply(
            lambda x: objective_function(
                forecaster,
                x.NumColdStarts,
                x.MemoryUsed,
                x.MemAllocated,
                x.SkipBlocks,
                app_level=True,
            ),
            axis=1,
        )

        vals = list(result_df.Metric.to_list())

        args["label"] = (
            "Invocation Count" if forecaster == "IceBreaker" else "Average Concurrency"
        )
        args["forecaster_num"] = forecast_num

        plot(vals, args=args)


def plot(data, args):
    data = [round(val, 6) for val in data]
    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))

    plt.set_cmap("cividis")
    linestyle = "-" if args["forecaster_num"] % 2 == 0 else "--"

    ax.plot(cdfx, cdfy, label=args["label"], linestyle=linestyle, color=colors[args["forecaster_num"]])
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2e6)

    ax.set_xlabel(args["x_label"])
    ax.set_ylabel(args["y_label"])
    ax.legend(title="Data Representation")
    plt.tight_layout()
    plt.savefig(args["file_name"])


if __name__ == "__main__":
    forecasters = ["FFT_10", "IceBreaker"]

    block_size = 504
    percentage = 100
    data_split = "train"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)
    plot_metric(forecasters, result_df)
    
    plt.close()
