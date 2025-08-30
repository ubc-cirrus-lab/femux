import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["pdf.fonttype"] = 42

data_dir = str(Path(__file__).parents[2] / "data"/ "azure") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots") + "/"

file_name = output_plots_dir + "faascache/faascache_comp.pdf"

ax = plt.figure(figsize=(6.8, 2.8), dpi=300, constrained_layout=True).add_subplot(1, 1, 1)
Faascache_points = {}
FeMux_points_0 = {}
FeMux_points_1 = {}

rum_vals = []
bar_labels = []

os.makedirs(output_plots_dir + "faascache/", exist_ok=True)

labelMapping = {
    "10 min keepalive": "10-min KA",
    "FaasCache 200.0GB": "FaasCache 200.0GB",
    "FaasCache 240.0GB": "FaasCache 240GB",
    "FaasCache 270.0GB": "FaasCache 270GB",
    "FaasCache 300.0GB": "FaasCache 300GB",
}

faascache_results_path = data_dir + "faascache/{}_results.pickle"
inv_exec_path = data_dir + "app_total_inv_exec_{}_days.pickle"

markers = [
    "$X$",
    "v",
    ".",
    "$FC$",
    "$IB$",
    "$10$",
    "$5$",
    "$1$",
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

COLD_START_DURATION = 0.808
WASTED_MEMORY_WEIGHT = 1 / 99.69
MS_IN_SEC = 1000
SECONDS_IN_HOUR = 3600


def plot_faascache(forecasters):
    args = dict()
    args["x_label"] = "Cold Start Count"
    args["y_label"] = "Wasted GB-seconds"
    args["line"] = False
    close = False 

    for forecaster_num, forecaster in enumerate(forecasters):
        forecaster_df = pd.read_pickle(faascache_results_path.format(forecaster))

        forecaster_num_cs = forecaster_df.NumColdStarts.sum()
        forecaster_wasted_mem = (
            forecaster_df.MemAllocated.sum() - forecaster_df.MemoryUsed.sum()
        )

        if forecaster == "IceBreaker":
            args["num"] = 4 
        elif forecaster == "10_min_keepalive":
            args["num"] = 5 
        elif forecaster == "5_min_keepalive":
            args["num"] = 6 
        elif forecaster == "Default_Knative":
            args["num"] = 7 
        print(forecaster)

        forecaster = forecaster.replace("_", " ")
        args["label"] = forecaster
        args["x"] = forecaster_num_cs
        args["y"] = forecaster_wasted_mem

        if forecaster == "FeMux":
            print(forecaster_num_cs, forecaster_wasted_mem)
            rum_vals.append(forecaster_num_cs * COLD_START_DURATION + forecaster_wasted_mem * WASTED_MEMORY_WEIGHT)
            bar_labels.append(args["label"])

        plot(args)

    faascache_df = pd.read_pickle(faascache_results_path.format("faascache"))

    cache_size_list = faascache_df["Cache Size"].to_list()

    for i in range(len(cache_size_list) - 1, -1, -1):
        cache_size = cache_size_list[i]
        faascache_num_cs = (
            faascache_df.iloc[i]["Misses"] + faascache_df.iloc[i]["Dropped"]
        )

        args["label"] = "FaasCache {}GB".format(cache_size / 1000)
        args["num"] = 3
        args["x"] = faascache_num_cs
        args["y"] = faascache_df.iloc[i]["Wasted Mem"]
        
        if "270" in args["label"]:
            print(faascache_num_cs, args["y"])
            rum_vals.append(faascache_num_cs * COLD_START_DURATION + args["y"] * WASTED_MEMORY_WEIGHT)
            bar_labels.append(args["label"])

        plot(args)


def plot(args):
    if "200.0GB" in args["label"]:
        return
    
    if "FaasCache" in args["label"]:
        Faascache_points[args["x"]] = args["y"]
    if "FeMux" in args["label"]:
        FeMux_points_0[args["x"]] = args["y"]

    if "log" in args and args["log"]:
        ax.set_yscale("log")

    marker = markers[args["num"]] if "num" in args else "o"
    if args["label"] in labelMapping.keys():
        args["label"] = labelMapping[args["label"]]
    scatter_size = 150 if "FaasCache" in args["label"] else 120
    ax.scatter(
        float(args["x"]),
        float(args["y"]),
        scatter_size,
        label=args["label"],
        marker=marker,
        zorder=1,
    )

    ax.set_xlabel(args["x_label"])
    ax.set_ylabel(args["y_label"])


if __name__ == "__main__":
    faascache_forecasters = ["10_min_keepalive", "5_min_keepalive", "Default_Knative", "IceBreaker"]

    plot_faascache(faascache_forecasters)
    
    connectingLineAlpha = 0.25

    ## final adjustments for ax
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    #ax.set_xlim(0, 3.05e5)
    #ax.set_ylim(0, 2.2e7)
    ax.yaxis.labelpad = -0.5
    ax.legend(
        ncol=2,
        columnspacing=0.2,
        borderpad=0.2,
        borderaxespad=0,
        handletextpad=0.2,
        loc="lower center",
    )
    ax.grid(True, which="both", axis="both", alpha=0.25)

    x_values = []
    y_values = []
    for key in sorted(Faascache_points.keys()):
        x_values.append(key)
        y_values.append(Faascache_points[key])
    ax.plot(
        x_values,
        y_values,
        color="black",
        linestyle="--",
        zorder=0,
        alpha=connectingLineAlpha,
    )

    x_values = []
    y_values = []
    for key in sorted(FeMux_points_0.keys()):
        x_values.append(key)
        y_values.append(FeMux_points_0[key])
    ax.plot(
        x_values,
        y_values,
        color="black",
        linestyle="-",
        zorder=0,
        alpha=connectingLineAlpha,
    )
    
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()