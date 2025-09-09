import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import gen_result_df, parse_size
from metric_calc import cold_start_seconds, wasted_GB_seconds

plt.rcParams["pdf.fonttype"] = 42

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"
save_path = str(Path(__file__).parents[1] / "output_plots") + "/"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

file_name = save_path + "sec_6_forecaster_comp_v2.pdf"

markers = [
    "$10$",
    "$5$",
    "$A$",
    "$E$",
    "$F$",
    "$H$",
    "$M$",
    "$S$",
    "$SW$",
    "$PS$",
    "^",
    "d",
    "v",
    "s",
    "*",
    "^",
    "d",
    "v",
    "s",
    "*",
    "^",
]

ax = plt.figure(figsize=(6, 3.3)).add_subplot(111)


def plot_agg(forecasters, result_df):
    args = dict()
    args["x_label"] = "Cold Start Seconds"
    args["y_label"] = "Wasted GB-seconds"
    args["log"] = False
    args["line"] = True

    # pareto frontier
    min_sum = 1000000000000000000000

    result_df.dropna(inplace=True)

    forecasters = sorted(forecasters)

    perfect_switching_df = pd.DataFrame()

    for forecast_num, forecaster in enumerate(forecasters):
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["ColdStartSec"] = result_df.apply(
            lambda x: cold_start_seconds(
                forecaster,
                x.NumColdStarts,
                x.MemoryUsed,
                x.MemAllocated,
                x.SkipBlocks,
                False,
            ),
            axis=1,
        )

        result_df["WastedMemTime"] = result_df.apply(
            lambda x: wasted_GB_seconds(
                forecaster,
                x.NumColdStarts,
                x.MemoryUsed,
                x.MemAllocated,
                x.SkipBlocks,
                False,
            ),
            axis=1,
        )

        if perfect_switching_df.empty:
            perfect_switching_df = result_df[["HashApp", "ColdStartSec", "WastedMemTime"]]
        else:
            perfect_switching_df = update_perfect_switching(perfect_switching_df, result_df)


        total_cs_time = result_df.ColdStartSec.apply(lambda x : sum(x)).sum()
        total_wm_time = result_df.WastedMemTime.apply(lambda x : sum(x)).sum()
        forecaster = "FeMux" if "default" in forecaster else forecaster

        args["label"] = forecaster

        args["x"] = np.round(total_cs_time)
        args["y"] = np.round(total_wm_time)

        args["num"] = forecast_num

        plot(args=args)

    print("Printing perfect") 
    args["label"] = "Perfect Switching"
    args["x"] = np.round(perfect_switching_df.ColdStartSec.apply(lambda x : sum(x)).sum())
    args["y"] = np.round(perfect_switching_df.WastedMemTime.apply(lambda x : sum(x)).sum())
    args["num"] = 9
    plot(args=args)


# scatter plot
def plot(args):
    plt.set_cmap("cividis")

    if "log" in args and args["log"]:
        plt.yscale("log")

    marker = markers[args["num"]] if "num" in args else "o"
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    arg_label = args["label"]
    if arg_label == "10_min_keepalive":
        arg_label = "10-min KA"
    if arg_label == "5_min_keepalive":
        arg_label = "5-min KA"
    elif arg_label == "FFT_10":
        arg_label = "FFT"
    elif arg_label == "MarkovChain_v3":
        arg_label = "Markov Chain"
    
    scatter_size = 150 if "FeMux" in arg_label else 100
    ax.scatter(
        float(args["x"]), float(args["y"]), scatter_size, label=arg_label, marker=marker, zorder=10
    )

    ax.set_xlabel(args["x_label"])
    ax.set_ylabel(args["y_label"])

    for axis in ["x", "y"]:
        if "%" in args["{}_label".format(axis)]:
            plt.ticklabel_format(style="plain")
        else:
            plt.ticklabel_format(style="sci", axis=axis, scilimits=(0, 0))

def update_perfect_switching(perfect_df, result_df):
    perfect_df.dropna(inplace=True)
    perfect_df = pd.merge(perfect_df, result_df[["HashApp", "ColdStartSec", "WastedMemTime"]], on="HashApp", suffixes=("_x", "_y"))
    perfect_df["ColdStartSec"] = perfect_df.apply(lambda x : take_min(np.array(x.ColdStartSec_x), np.array(x.ColdStartSec_y)), axis=1)
    perfect_df["WastedMemTime"] = perfect_df.apply(lambda x : take_min(np.array(x.WastedMemTime_x), np.array(x.WastedMemTime_y)), axis=1)

    return perfect_df[["HashApp", "ColdStartSec", "WastedMemTime"]]

def take_min(x, y):
    return [min(x[i], y[i]) for i in range(len(x))]

if __name__ == "__main__":
    forecasters = [
        "5_min_keepalive",
        "10_min_keepalive",
        "AR_10",
        #"ExpSmoothing",
        "MarkovChain_v3",
        #"SETAR",
        #"Holt",
        "FFT_10",
        #"Oracle",
        "default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics"
    ]

    block_size = 504
    percentage = 100
    data_split = "test"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)
    plot_agg(forecasters, result_df)

    x = np.linspace(0, 5e6, 100)
    y = -99.7 * x
    for i in range(1, 12):
        ax.plot(x, y + i * 1e8, color="black", zorder=-10, alpha=0.08)

    ax.annotate(
        "",
        xytext=(1.22e6, 2.2e8),
        xy=(0.8e6, 0.84e8),
        horizontalalignment="center",
        arrowprops=dict(arrowstyle="fancy", color="black"),
    )
    ax.annotate(
        "Optimal",
        xy=(1.2e6, 2.5e8),
        xytext=(0.8e6, 0.34e8),
        horizontalalignment="center",
    )

    ax.set_xlim(0, 5e6)
    ax.set_ylim(0, 6e8)

    plt.legend(bbox_to_anchor=(1.01, 0.83))
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")

    plt.close()