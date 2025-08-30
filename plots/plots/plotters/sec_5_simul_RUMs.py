import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from data_processing import gen_result_df
from metric_calc import cold_start_seconds, wasted_GB_seconds
from matplotlib.lines import Line2D

plt.rcParams["pdf.fonttype"] = 42

data_dir = str(Path(__file__).parents[2] / "data") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots") + "/"

ax = plt.figure(figsize=(4.8, 2.8), dpi=300, constrained_layout=True).subplots(1)
xPoints = defaultdict(list)
yPoints = defaultdict(list)


TOTAL_NUM_DAYS = 12
NUM_TRAINING_DAYS = 7
NUM_AQUATOPE_INPUT_STEPS = 48
BLOCK_SIZE = 504
STATIC_KEEPALIVE_WINDOW = 10
MINUTES_PER_DAY = 1440
START_BLOCK = 21
START_INDEX = BLOCK_SIZE * 21
END_INDEX = BLOCK_SIZE * 34


inv_exec_path = data_dir + "app_total_inv_exec_{}_days.pickle"
forecaster_template = (
    "{}_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics"
)

rum_vals = []
bar_labels = []

COLD_START_DURATION = 0.808
WASTED_MEMORY_WEIGHT = 1 / 99.69
MS_IN_SEC = 1000
SECONDS_IN_HOUR = 3600
FRAC_CS = COLD_START_DURATION / 60

colours = {"FeMux-CS": "tab:blue", "FeMux": "tab:orange", "FeMux-Mem": "tab:green"}


def plot_final(forecasters, data_desc):
    result_df = gen_result_df(forecasters, data_desc)

    # pareto frontier
    inv_df = pd.read_pickle(inv_exec_path.format(TOTAL_NUM_DAYS))
    result_df = result_df.merge(inv_df, on="HashApp")
    result_df["NumInvocations"] = result_df["InvocationsPerMin"].apply(lambda x: sum(x))
    result_df.sort_values("NumInvocations", inplace=True, ascending=False)

    long_cs_df = result_df.iloc[::10, :]
    short_cs_df = result_df[~result_df.HashApp.isin(long_cs_df.HashApp)]
    print(long_cs_df)
    print(short_cs_df)

    plot_simul_rums(
        [forecaster_template.format("4_cs"), forecaster_template.format("default")],
        long_cs_df,
        "P",
    )
    plot_simul_rums(
        [forecaster_template.format("4_cs"), forecaster_template.format("default")],
        short_cs_df,
        "R",
    )


def plot_simul_rums(forecasters, result_df, data_split):
    args = dict()
    args["x_label"] = "Average Cold Start Seconds Per App"
    args["y_label"] = "Wasted GB-seconds"
    args["log"] = False
    args["related_work"] = "Aquatope"

    for forecast_num, forecaster in enumerate(forecasters):
        print(forecaster)

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

        if "4_wm" in forecaster:
            forecaster = "FeMux-Mem"
        elif "4_cs" in forecaster:
            forecaster = "FeMux-CS"
        elif "default" in forecaster:
            forecaster = "FeMux"

        args["label"] = data_split
        args["forecaster"] = forecaster
        args["x"] = np.round(total_cs_time) / len(result_df)
        args["y"] = np.round(total_wm_time)
        print(args["x"], args["y"])
        xPoints[data_split].append(args["x"])
        yPoints[data_split].append(args["y"])

        plot_simul(args=args)


def plot_simul(args):
    if "log" in args and args["log"]:
        ax.set_yscale("log")

    scatter_size = 150

    ax.scatter(
        float(args["x"]),
        float(args["y"]),
        scatter_size,
        marker="${}$".format(args["label"]),
        color=colours[args["forecaster"]],
        zorder=1,
    )

    ax.set_xlabel(args["x_label"])
    ax.set_ylabel(args["y_label"])


if __name__ == "__main__":
    data_desc = "504_100_percent_test"
    file_desc = "FeMux_test"

    forecasters = [
        "4_cs_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
        "default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
        "4_wm_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics",
    ]

    plot_final(forecasters, data_desc)

    ax.plot(
        [xPoints["P"][0], xPoints["R"][0]],
        [yPoints["P"][0], yPoints["R"][0]],
        color="black",
        linestyle="--",
        zorder=0,
        alpha=0.25,
        linewidth=2,
    )
    ax.plot(
        [xPoints["P"][1], xPoints["R"][1]],
        [yPoints["P"][1], yPoints["R"][1]],
        color="black",
        linestyle="--",
        zorder=0,
        alpha=0.25,
        linewidth=2,
    )

    ax.plot(
        [xPoints["P"][0], xPoints["R"][1]],
        [yPoints["P"][0], yPoints["R"][1]],
        color="black",
        linestyle="-",
        zorder=0,
        alpha=0.5,
        linewidth=2,
    )
    
    # plot an arrow from [xPoints["P"][0], yPoints["P"][0]] to [xPoints["R"][0], yPoints["R"][0]]
    # the arrow should be gray, dashed, with two heads, and a width of 0.5
    ax.annotate(
        "1-Tiered",
        xy=(xPoints["R"][0], yPoints["R"][0]),
        xytext=(1.05*(xPoints["R"][0]+xPoints["P"][0])/2, (yPoints["R"][0]+yPoints["P"][0])/2),
        rotation=-82,
        ha="center",
        va="center",
    )
    ax.annotate(
        "1-Tiered",
        xy=(xPoints["R"][0], yPoints["R"][0]),
        xytext=(1.04*(xPoints["R"][1]+xPoints["P"][1])/2, (yPoints["R"][1]+yPoints["P"][1])/2),
        rotation=-92,
        ha="center",
        va="center",
    )
    ax.annotate(
        "2-Tiered",
        xy=(xPoints["R"][1], yPoints["R"][1]),
        xytext=((xPoints["P"][0]+xPoints["R"][1])/2, 0.8*(yPoints["P"][0]+yPoints["R"][1])/2),
        rotation=30,
    )

    connectingLineAlpha = 0.25

    ## final adjustments for ax

    ## final adjustments for axs[1]

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                marker="$P$",
                linestyle="None",
                markersize=10,
                label="Premium Apps",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="$R$",
                linestyle="None",
                markersize=10,
                label="Regular Apps",
            ),
            Line2D(
                [],
                [],
                color="blue",
                marker=".",
                linestyle="None",
                markersize=10,
                label="FeMux-CS",
            ),
            Line2D(
                [],
                [],
                color="orange",
                marker=".",
                linestyle="None",
                markersize=10,
                label="FeMux",
            ),
        ]
    )
    ax.set_ylim(0, 450000000)
    ax.set_xlim(0, 400)
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.yaxis.labelpad = 0

    plt.tight_layout()
    plt.savefig(output_plots_dir + "sec_5_simul_RUMs.pdf", bbox_inches="tight")
    plt.close()
