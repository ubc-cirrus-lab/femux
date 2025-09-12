import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_processing import gen_result_df
from metric_calc import cold_start_seconds, wasted_GB_seconds

plt.rcParams["pdf.fonttype"] = 42

data_dir = str(Path(__file__).parents[2] / "data") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots") + "/"

file_name = output_plots_dir + "sec_5_prior_work_{}.pdf"
faascache_results_path = data_dir + "faascache_data/{}_results.pickle"


axs = plt.figure(figsize=(11.2, 3), dpi=300, constrained_layout=True).subplots(1, 3)

Faascache_points = {}
FeMux_points_0 = {}

labelMapping = {
    "10 min keepalive": "10-min KA",
    "FaasCache 200.0GB": "FaasCache 200.0GB",
    "FaasCache 240.0GB": "FaasCache 240GB",
    "FaasCache 270.0GB": "FaasCache 270GB",
    "FaasCache 300.0GB": "FaasCache 300GB",
}


TOTAL_NUM_DAYS = 12
NUM_TRAINING_DAYS = 7
NUM_AQUATOPE_INPUT_STEPS = 48
BLOCK_SIZE = 504
STATIC_KEEPALIVE_WINDOW = 10
MINUTES_PER_DAY = 1440
START_BLOCK = 21
START_INDEX = BLOCK_SIZE * 21
END_INDEX = BLOCK_SIZE * 34
WASTED_MEMORY_WEIGHT = 1 / 99.69
MS_IN_SEC = 1000
SECONDS_IN_HOUR = 3600
COLD_START_DURATION = 0.808


CHEAP_NODE_COST = 0.0084 / SECONDS_IN_HOUR


inv_exec_path = data_dir + "app_total_inv_exec_{}_days.pickle"

rum_vals = []
bar_labels = []

markers = [
    "$FM$",
    "v",
    ".",
    "$FC$",
    "$IB$",
    "$AQ$",
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


def plot_final(forecasters, data_desc):
    #plot_faascache(axs[0])
    plot_icebreaker(forecasters + ["10_min_keepalive"], data_desc, axs[1])
    plot_aquatope(forecasters + ["10_min_keepalive"], data_desc, axs[2])


def plot_aquatope(forecasters, data_desc, ax):
    global FeMux_points_0
    FeMux_points_0 = {}
    result_df = gen_result_df(forecasters, data_desc, gen_skip=False)

    args = dict()
    args["x_label"] = "Cold Starts (%)"
    args["y_label"] = "Normalized Memory Allocation (%)"  # Wasted GB-seconds"
    args["log"] = False
    args["line"] = False
    args["related_work"] = "Aquatope"
    close = False

    # pareto frontier
    min_sum = 1000000000000000000000

    forecasters.remove("10_min_keepalive")
    static_ka_mem_alloc = result_df.MemAllocated.apply(
        lambda x: sum(x["10_min_keepalive"][START_BLOCK:])
    ).sum()
    num_invocations = total_num_invocations(result_df)
    print(num_invocations)

    for forecast_num, forecaster in enumerate(forecasters):
        print(forecaster)
        if forecast_num == len(forecasters) - 1:
            close = True

        cs_count = result_df.NumColdStarts.apply(
            lambda x: (
                x[forecaster]
                if forecaster == "Aquatope"
                else sum(x[forecaster][START_BLOCK:])
            )
        ).sum()
        mem_alloc = result_df.MemAllocated.apply(
            lambda x: (
                x[forecaster]
                if forecaster == "Aquatope"
                else sum(x[forecaster][START_BLOCK:])
            )
        ).sum()
        mem_used = result_df.MemoryUsed.apply(
            lambda x: (
                x[forecaster]
                if forecaster == "Aquatope"
                else sum(x[forecaster][START_BLOCK:])
            )
        ).sum()

        args["num"] = 0
        if "4_wm" in forecaster:
            forecaster = "FeMux-Mem"
        elif "16_wm" in forecaster:
            forecaster = "FeMux-Mem(16)"
        elif "4_cs" in forecaster:
            forecaster = "FeMux-CS"
        elif "default" in forecaster:
            forecaster = "FeMux"
        else:
            args["num"] = 5

        forecaster = forecaster.replace("_", " ")

        args["label"] = forecaster
        args["x"] = np.round(100 * cs_count / num_invocations, 2)
        args["y"] = np.round(mem_alloc / static_ka_mem_alloc * 100, 2)
        print(args["x"], args["y"], cs_count)

        if forecaster == "FeMux" or forecaster == "Aquatope":
            rum_vals.append(
                cs_count * COLD_START_DURATION
                + (mem_alloc - mem_used) * WASTED_MEMORY_WEIGHT
            )
            bar_labels.append(forecaster)

        plot_aq(args, ax)

    plot_line(ax)


def total_num_invocations(aquatope_df):
    inv_df = pd.read_pickle(inv_exec_path.format(TOTAL_NUM_DAYS))
    inv_df = inv_df[inv_df["HashApp"].isin(aquatope_df["HashApp"])]
    inv_df["NumInvocations"] = inv_df["InvocationsPerMin"].apply(
        lambda x: x[START_INDEX:END_INDEX].sum()
    )

    return inv_df["NumInvocations"].sum()


def plot_aq(args, ax):
    if "FeMux" in args["label"]:
        FeMux_points_0[args["x"]] = args["y"]

    if "log" in args and args["log"]:
        ax.set_yscale("log")

    marker = markers[args["num"]] if "num" in args else "o"
    scatter_size = 150
    if "Aquatope" in args["label"]:
        ax.scatter(
            float(args["x"]),
            float(args["y"]),
            scatter_size,
            label="Aquatope's LSTM",
            marker=marker,
            zorder=1,
            color="midnightblue",
        )
    else:
        ax.scatter(
            float(args["x"]),
            float(args["y"]),
            scatter_size,
            marker=marker,
            zorder=1,
        )

    ax.set_xlabel(args["x_label"])
    ax.set_ylabel(args["y_label"])


def plot_icebreaker(forecasters, data_desc, ax):
    global FeMux_points_0
    FeMux_points_0 = {}
    result_df = gen_result_df(forecasters, data_desc)
    invocation_df = pd.read_pickle(inv_exec_path.format(12))
    result_df = result_df.merge(invocation_df, on="HashApp", how="left")

    args = dict()
    args["x_label"] = "Normalized Service Time (%)"
    args["y_label"] = "Normalized Keep-alive Cost (%)"  # Wasted GB-seconds"
    args["log"] = False
    args["line"] = False
    args["related_work"] = "IceBreaker"
    close = False

    # pareto frontier
    min_sum = 1000000000000000000000

    total_exec_time = result_df.TotalInvocations.sum() / MS_IN_SEC

    # 10_min keepalive baseline
    result_df["ColdStartSec"] = result_df.apply(
        lambda x: cold_start_seconds(
            "10_min_keepalive",
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
            "10_min_keepalive",
            x.NumColdStarts,
            x.MemoryUsed,
            x.MemAllocated,
            x.SkipBlocks,
            True,
        ),
        axis=1,
    )

    static_service_time = result_df.ColdStartSec.sum() + total_exec_time
    static_ka_cost = result_df.WastedMemTime.sum() * CHEAP_NODE_COST

    forecasters.remove("10_min_keepalive")

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

        total_service_time = (
            100 * (result_df.ColdStartSec.sum() + total_exec_time) / static_service_time
        )
        total_ka_cost = (
            100 * (result_df.WastedMemTime.sum() * CHEAP_NODE_COST) / static_ka_cost
        )

        args["num"] = 0
        if "4_wm" in forecaster:
            forecaster = "FeMux-Mem"
        elif "16_wm" in forecaster:
            forecaster = "FeMux-Mem(16)"
        elif "4_cs" in forecaster:
            forecaster = "FeMux-CS"
        elif "default" in forecaster:
            forecaster = "FeMux"
        else:
            args["num"] = 4

        forecaster = forecaster.replace("_", " ")
        args["label"] = forecaster
        args["x"] = np.round(total_service_time)
        args["y"] = np.round(total_ka_cost)

        print("Cold start time:" + str(args["x"]))
        print("Wasted Memory Time: " + str(args["y"]))

        if forecaster == "FeMux" or forecaster == "IceBreaker":
            rum_vals.append(
                result_df.ColdStartSec.sum()
                + result_df.WastedMemTime.sum() * WASTED_MEMORY_WEIGHT
            )
            bar_labels.append(forecaster)

        if (args["x"] + args["y"] / 100) < min_sum:
            args["x_best"] = args["x"]
            args["y_best"] = args["y"]
            min_sum = args["x"] + args["y"] / 100

        print(forecaster)
        print(print(args["x"] + args["y"]))

        plot_ib(args, ax)

    plot_line(ax)


def plot_ib(args, ax):
    if "FeMux" in args["label"]:
        FeMux_points_0[args["x"]] = args["y"]

    if "log" in args and args["log"]:
        ax.set_yscale("log")

    marker = markers[args["num"]] if "num" in args else "o"
    scatter_size = 150 if "FaasCache" in args["label"] else 120
    ax.scatter(
        float(args["x"]),
        float(args["y"]),
        scatter_size,
        label=None if "FeMux" in args["label"] else "IceBreaker's FFT",
        marker=marker,
        zorder=1,
    )

    ax.set_xlabel(args["x_label"])
    ax.set_ylabel(args["y_label"])


def plot_faascache(ax):
    faascache_forecasters = ["4_cs", "default", "4_wm"]
    args = dict()
    args["x_label"] = "Cold Start Count"
    args["y_label"] = "Wasted GB-seconds"
    args["line"] = False

    for forecaster_num, forecaster in enumerate(faascache_forecasters):
        forecaster_df = pd.read_pickle(faascache_results_path.format(forecaster))

        forecaster_num_cs = forecaster_df.NumColdStarts.sum()
        forecaster_wasted_mem = (
            forecaster_df.MemAllocated.sum() - forecaster_df.MemoryUsed.sum()
        )

        args["num"] = 0

        if "wm" in forecaster:
            forecaster = "FeMux-Mem"
        elif "4_cs" in forecaster:
            forecaster = "FeMux-CS"
        elif "default" in forecaster:
            forecaster = "FeMux"
        else:
            args["num"] = forecaster_num + 1

        forecaster = forecaster.replace("_", " ")
        args["label"] = forecaster
        args["x"] = forecaster_num_cs
        args["y"] = forecaster_wasted_mem

        if forecaster == "FeMux":
            print(forecaster_num_cs, forecaster_wasted_mem)
            rum_vals.append(
                forecaster_num_cs * COLD_START_DURATION
                + forecaster_wasted_mem * WASTED_MEMORY_WEIGHT
            )
            bar_labels.append(args["label"])

        plot_fc(args, ax)

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
            rum_vals.append(
                faascache_num_cs * COLD_START_DURATION
                + args["y"] * WASTED_MEMORY_WEIGHT
            )
            bar_labels.append(args["label"])

        plot_fc(args, ax)

    plot_line(ax)


def plot_fc(args, ax):
    print(args)
    if "200.0GB" in args["label"]:
        return

    if "keepalive" in args["label"]:
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


def plot_line(ax):
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
        alpha=0.25,
    )


if __name__ == "__main__":
    data_desc = "504_100_percent_test"
    file_desc = "FeMux_test"

    forecasters = [
        "4_cs_kmeans_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
        "default_kmeans_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
        "4_wm_kmeans_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
    ]

    plot_final(forecasters, data_desc)

    ## Faascache
    #axs[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    #axs[0].set_xlim(0, 3.05e5)
    #axs[0].set_ylim(0, 2.2e7)
    #axs[0].yaxis.labelpad = 0
    #axs[0].legend(
        #ncol=2,
        #columnspacing=0.2,
        #borderpad=0.2,
        #borderaxespad=0,
        #handletextpad=0.2,
        #loc="lower center",
        #fontsize=9,
    #)
    #axs[0].grid(True, which="both", axis="both", alpha=0.25)

    #x_values = []
    #y_values = []
    #for key in sorted(Faascache_points.keys()):
        #x_values.append(key)
        #y_values.append(Faascache_points[key])
    #axs[0].plot(
        #x_values,
        #y_values,
        #color="black",
        #linestyle="--",
        #zorder=0,
        #alpha=0.25,
    #)

    # IceBreaker
    axs[1].set_xlim(0, 300)
    axs[1].set_ylim(0, 100)
    axs[1].grid(True, which="both", axis="both", alpha=0.25)
    axs[1].yaxis.labelpad = 0
    axs[1].legend()

    ## Aquatope
    #axs[2].set_xlim(0, 0.6)
    #axs[2].set_ylim(0, 250)
    #axs[2].grid(True, which="both", axis="both", alpha=0.25)
    #axs[2].yaxis.labelpad = 0
    #axs[2].legend()

    plt.tight_layout()
    plt.savefig(file_name.format("all"), bbox_inches="tight")
    plt.close()
