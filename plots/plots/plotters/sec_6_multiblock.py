import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from itertools import chain, combinations
from metric_calc import objective_function
from data_processing import gen_result_df, parse_size

sys.path.append("../../")
# from plots.plotters.plotter import plot

plt.rcParams["pdf.fonttype"] = 42

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots") + "/"

hashapp_list_path = data_dir + "train_test_split/{}_{}_apps.pickle"
cs_wm_forecaster_path = data_dir + "/_cold_starts_wasted_mem.pickle"
save_path = output_plots_dir + "../../"
inv_exec_path = data_dir + "app_total_inv_exec_{}_days.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

markers = [
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


def plot_multi(forecasters, data_desc, block_sizes):
    result_df = pd.DataFrame()
    agg_forecasters = []

    for forecaster in forecasters:
        for block_size in block_sizes:

            data_path = cs_wm_forecaster_path.replace(
                "/_cold_starts",
                "/{}/{}_cold_starts".format(data_desc.format(block_size), forecaster),
            )
            forecaster_df = pd.read_pickle(data_path)
            forecaster_df = forecaster_df[
                [
                    "HashApp",
                    "NumColdStarts",
                    "NumInvocations",
                    "MemAllocated",
                    "MemoryUsed",
                ]
            ]

            forecaster_df["SumColdStarts"] = forecaster_df.NumColdStarts.apply(
                lambda x: sum(x)
            )
            forecaster_df = forecaster_df[forecaster_df.SumColdStarts > 0]
            forecaster_df.drop(["SumColdStarts"], inplace=True, axis=1)

            result_df = update_result_df(
                result_df, forecaster_df, forecaster + str(block_size)
            )
            agg_forecasters.append(forecaster + str(block_size))

    for size in ["small", "medium", "large", "all"]:
        os.makedirs(save_path + "{}/{}/".format(size, "multiblock"), exist_ok=True)

    invocation_df = pd.read_pickle(inv_exec_path.format(12))
    result_df = result_df.merge(invocation_df, on="HashApp", how="left")

    print(agg_forecasters)
    for size in sizes:
        plot_agg(agg_forecasters, file_desc, size, result_df)


def plot(data, bins, args, close=False):
    if "forecaster" in args or "ColName" in args:
        count, bins_count = np.histogram(data, bins=bins)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

    if "x" in args:
        # scatter plot
        plt.set_cmap("cividis")
        data = [round(val, 6) for val in data]
        if "log" in args and args["log"]:
            plt.yscale("log")

        marker = markers[args["num"]] if "num" in args else "o"
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.scatter(
            int(args["x"]),
            int(args["y"]),
            80,
            label=args["label"],
            marker=marker,
            zorder=1,
        )

        plt.xlabel(args["x_label"])
        plt.ylabel(args["y_label"])
        plt.legend()
        for axis in ["x", "y"]:
            if "%" in args["{}_label".format(axis)]:
                plt.ticklabel_format(style="plain")
            else:
                plt.ticklabel_format(style="sci", axis=axis, scilimits=(0, 0))

        plt.savefig(args["file_name"] + ".pdf", bbox_inches="tight")

    if close:
        plt.close()


def update_result_df(result_df, forecast_df, forecaster):
    """For each application, aggregate all forecaster data into a dict for each app"""

    if result_df.empty:
        result_df = forecast_df

        for col in result_df.columns:
            if col != "HashApp":
                result_df[col] = result_df[col].apply(lambda x: {forecaster: x})

        return result_df

    result_df = result_df.merge(
        forecast_df, on="HashApp", how="left", suffixes=(None, "_y")
    )
    dropped_df = result_df.dropna()

    if len(dropped_df) < len(result_df):
        print(
            "Dropped {} apps due to missing values in {}".format(
                len(result_df) - len(dropped_df), forecaster
            )
        )
        result_df = dropped_df

    for col in result_df.columns:
        if col == "HashApp" or col.endswith("_y"):
            continue
        result_df[col] = result_df.apply(
            lambda x: update_dict(x[col], x[col + "_y"], forecaster), axis=1
        )
        del result_df[col + "_y"]

    return result_df


def update_dict(result_dict, new_vals, forecaster):
    result_dict[forecaster] = new_vals
    return result_dict


# modify this for the paper plot
def plot_agg(forecasters, file_desc, size, result_df):
    # result_df = parse_size(result_df, "test", "small")
    # result_df = parse_size(result_df, "test", "medium")
    # result_df = parse_size(result_df, "test", "large")

    cs_420 = (
        result_df["NumColdStarts"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics420"
            ].sum()
        )
        .sum()
    )
    cs_504 = (
        result_df["NumColdStarts"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics504"
            ].sum()
        )
        .sum()
    )
    cs_720 = (
        result_df["NumColdStarts"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics720"
            ].sum()
        )
        .sum()
    )
    cs_1440 = (
        result_df["NumColdStarts"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics1440"
            ].sum()
        )
        .sum()
    )

    mw_420 = (
        result_df["MemAllocated"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics420"
            ].sum()
        )
        .sum()
        - result_df["MemoryUsed"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics420"
            ].sum()
        )
        .sum()
    )

    mw_504 = (
        result_df["MemAllocated"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics504"
            ].sum()
        )
        .sum()
        - result_df["MemoryUsed"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics504"
            ].sum()
        )
        .sum()
    )

    mw_720 = (
        result_df["MemAllocated"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics720"
            ].sum()
        )
        .sum()
        - result_df["MemoryUsed"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics720"
            ].sum()
        )
        .sum()
    )

    mw_1440 = (
        result_df["MemAllocated"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics1440"
            ].sum()
        )
        .sum()
        - result_df["MemoryUsed"]
        .apply(
            lambda x: x[
                "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics1440"
            ].sum()
        )
        .sum()
    )

    adjustment_weight = [0.99248, 1, 1]

    cold_starts = [cs_420, cs_720, cs_1440]
    mem_wasted = [mw_420, mw_720, mw_1440]

    for i in range(len(adjustment_weight)):
        cold_starts[i] = cold_starts[i] * adjustment_weight[i]
        mem_wasted[i] = mem_wasted[i] * adjustment_weight[i]

    rum = [
        cold_starts[i] * 0.808 + mem_wasted[i] * 1 / 99.69
        for i in range(len(cold_starts))
    ]

    # plot cold starts and wasted memory in a single plot with two y axes
    axs = plt.figure(figsize=(4.5, 2.25)).subplots(1, 2)
    ax1 = axs[0]
    ax2 = axs[1]

    x = [420, 720, 1440]
    x = [int(v / 60) for v in x]

    for ax in axs:
        ax.set_xlabel("Block Size (hr)")
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        # grid lines
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    ax1.set_ylabel("Cold Starts")
    # ax1.set_ylabel("RUM")

    ax1.plot(x, cold_starts, color="tab:red", marker="o")
    ax1.tick_params(axis="y")  # , labelcolor="tab:red"

    ax1.set_ylim(0, 2.8e6)

    ax2.set_ylabel("Wasted Memory (GB)")
    ax2.plot(x, mem_wasted, color="tab:blue", marker="^")
    ax2.tick_params(axis="y")
    ax2.set_ylim(0, 4e8)

    plt.tight_layout()
    plt.savefig("../output_plots/block_size_sensitivity.pdf")


def cold_start_seconds(forecaster, num_cold_starts):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []
    num_orig_blocks = len(num_cold_starts[forecaster])

    if num_orig_blocks == 40:
        num_blocks = 38
    elif num_orig_blocks == 33:
        num_blocks = 31
    elif num_orig_blocks == 23:
        num_blocks = 22
    else:
        num_blocks = 11

    for block_index in range(num_blocks):
        obj_vals.append(num_cold_starts[forecaster][block_index] * COLD_START_DURATION)

    return sum(obj_vals)


def wasted_GB_seconds(forecaster, mem_used, mem_allocated):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []

    num_orig_blocks = len(mem_allocated[forecaster])

    if num_orig_blocks == 40:
        num_blocks = 38
    elif num_orig_blocks == 33:
        num_blocks = 31
    elif num_orig_blocks == 23:
        num_blocks = 22
    else:
        num_blocks = 11

    for block_index in range(num_blocks):  # len(mem_allocated[forecaster]) - offset):
        obj_vals.append(
            mem_allocated[forecaster][block_index] - mem_used[forecaster][block_index]
        )

    return sum(obj_vals)


if __name__ == "__main__":
    forecasters = [
        "default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics"
    ]

    file_desc = "multiblock_markov_v3"
    block_sizes = [420, 504, 720, 1440]
    percentage = 100
    data_desc = "{}_100_percent_test"

    sizes = ["all"]

    result_df = plot_multi(forecasters, data_desc, block_sizes)
