import numpy as np
import sys
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

ax = plt.figure(figsize=(4, 3)).add_subplot(111)

markers = [
    "$D$",
    "$H$",
    "$L$",
    "$S$",
    "$DH$",
    "$DL$",
    "$DS$",
    "$LS$",
    "$LH$",
    "$SH$",
    "$DLH$",
    "$DLS$",
    "$DSH$",
    "$LSH$",
    "$DLSH$",
    "d",
    "v",
    "s",
    "*",
    "^",
]


def plot_agg(forecasters, result_df):
    args = dict()
    args["x_label"] = "Cold Start Seconds"
    args["y_label"] = "Wasted GB-seconds"
    args["file_name"] = save_path + "sec_6_feature_comp.pdf"
    args["log"] = False
    args["line"] = True

    # pareto frontier
    min_sum = 1000000000000000000000

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
                True,
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
                True,
            ),
            axis=1,
        )

        total_cs_time = sum(result_df.ColdStartSec.to_list())
        total_wm_time = sum(result_df.WastedMemTime.to_list())

        forecaster = forecaster.replace("default_markov_v3_StandardScaler", "")
        forecaster = forecaster.replace("femux", "")
        forecaster = forecaster.replace("Density", "D")
        forecaster = forecaster.replace("Linearity", "L")
        forecaster = forecaster.replace("Stationarity", "S")
        forecaster = forecaster.replace("Harmonics", "H")
        forecaster = forecaster.replace("_", " ")

        args["label"] = forecaster

        args["x"] = np.round(total_cs_time)
        args["y"] = np.round(total_wm_time)

        args["num"] = forecast_num

        plot(args=args)


# create a colormap with seven colors
colors = ["#e41a1c", "#f781bf", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#377eb8"]

# scatter plot
def plot(args):
    # plt.set_cmap("cividis")
    plt.set_cmap("winter")

    # if "log" in args and args["log"]:
    #     plt.yscale("log")

    marker = markers[args["num"]] if "num" in args else "o"

    if len(args["label"]) == 3:
        if "D" in args["label"]:
            new_label = "Density"
        elif "L" in args["label"]:
            new_label = "Linearity"
        elif "S" in args["label"]:
            new_label = "Stationarity"
        elif "H" in args["label"]:
            new_label = "Harmonics"
        ax.scatter(
            int(args["x"]),
            int(args["y"]),
            (len(args["label"]) - 1) * 30,
            label=new_label,
            marker=marker,
            color=colors[len(args["label"]) - 3],
            zorder=1,
        )
    else:
        ax.scatter(
            int(args["x"]),
            int(args["y"]),
            (len(args["label"]) - 2) * 50,
            marker=marker,
            color=colors[len(args["label"]) - 3],
            zorder=1,
        )
    ax.scatter(
        int(args["x"]),
        int(args["y"]),
        4,
        marker=".",
        color="black",
        zorder=1,
        alpha=0.5,
    )
    # plt.legend()
    # for axis in ["x", "y"]:
    #     if "%" in args["{}_label".format(axis)]:
    #         plt.ticklabel_format(style="plain")
    #     else:
    #         plt.ticklabel_format(style="sci", axis=axis, scilimits=(0, 0))


if __name__ == "__main__":
    forecasters = [
        "default_markov_v3_StandardScaler_femux_Density",
        "default_markov_v3_StandardScaler_femux_Harmonics",
        "default_markov_v3_StandardScaler_femux_Linearity",
        "default_markov_v3_StandardScaler_femux_Stationarity",
        "default_markov_v3_StandardScaler_femux_Density_Harmonics",
        "default_markov_v3_StandardScaler_femux_Density_Linearity",
        "default_markov_v3_StandardScaler_femux_Density_Stationarity",
        "default_markov_v3_StandardScaler_femux_Linearity_Stationarity",
        "default_markov_v3_StandardScaler_femux_Linearity_Harmonics",
        "default_markov_v3_StandardScaler_femux_Stationarity_Harmonics",
        "default_markov_v3_StandardScaler_femux_Density_Linearity_Harmonics",
        "default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity",
        "default_markov_v3_StandardScaler_femux_Density_Stationarity_Harmonics",
        "default_markov_v3_StandardScaler_femux_Linearity_Stationarity_Harmonics",
        "default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
    ]

    block_size = 504
    percentage = 100
    data_split = "test"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)
    plot_agg(forecasters, result_df)

    # ax.set_yscale("log")
    ax.set_xlabel("Cold Start Seconds")
    ax.set_ylabel("Wasted GB-seconds")

    ax.set_xlim(2e6, 3.6e6)
    ax.set_ylim(2e8, 3e8)

    x = np.array([1.5e6, 4e6])
    y = -99.7 * x + 2.6e8 + 1.5e6 * 99.7
    for i in range(1, 16):
        ax.plot(x, y + i * 0.2e8, color="black", zorder=-10, alpha=0.08)

    ax.legend()

    plt.savefig(save_path + "sec_6_feature_comp.pdf", bbox_inches="tight")
    plt.close()
