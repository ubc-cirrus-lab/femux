import matplotlib.pyplot as plt
from data_processing import gen_result_df, parse_size
from metric_calc import objective_function
import pandas as pd
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"
save_path = str(Path(__file__).parents[1] / "output_plots") + "/"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

df_data = {"Forecaster": [], "Invocations": [], "RUM": []}

name_map = {"Adaptive_FFT_10": "FFT", "AR": "AR"}


def plot_metric(forecasters, result_df, size):
    args = dict()
    args["x_label"] = "RUM"
    args["y_label"] = "Fraction of Apps"
    args["log"] = True
    args["title"] = size

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

        vals = result_df.Metric.to_list()

        if forecaster in name_map.keys():
            args["label"] = name_map[forecaster]
            df_data["Forecaster"].append(name_map[forecaster])
        else:
            args["label"] = forecaster
            df_data["Forecaster"].append(forecaster)

        args["forecaster_num"] = forecast_num

        df_data["Invocations"].append(size)
        df_data["RUM"].append(sum(vals))


if __name__ == "__main__":
    forecasters = ["AR", "Adaptive_FFT_10"]

    block_size = 504
    percentage = 100
    data_split = "train"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)

    small_result_df = parse_size(result_df, data_split, "small")
    medium_result_df = parse_size(result_df, data_split, "medium")
    large_result_df = parse_size(result_df, data_split, "large")

    plot_metric(forecasters, small_result_df, "<1 Million")
    plot_metric(forecasters, medium_result_df, "1M-100M")
    plot_metric(forecasters, large_result_df, ">100 Million")

    df = pd.DataFrame(df_data)

    fig, axs = plt.subplots(
        1, 2, figsize=(5.25, 2.75), dpi=300, gridspec_kw={"width_ratios": [2, 1]}
    )

    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    sns.barplot(
        x="Invocations", y="RUM", hue="Forecaster", data=df, ax=axs[0], width=0.8
    )

    winners = {
        "<1 Million": "Adaptive_FFT_10",
        "1M-100M": "AR",
        ">100 Million": "AR",
    }

    i = 0
    win = {"x": [], "y": []}
    for key in winners:
        winner = winners[key]
        d = forecasters.index(winner)
        y = df[(df["Forecaster"] == name_map[winner]) & (df["Invocations"] == key)][
            "RUM"
        ].values[0]
        win["x"].append(i + d * 0.4 - 0.2)
        win["y"].append(y)
        i += 1

    print(df)

    axs[0].scatter(
        win["x"],
        win["y"],
        marker="*",
        s=100,
        color="black",
        zorder=3,
        label="Optimal Forecaster",
    )

    # axs[0].set_xlabel("")
    # ax.set_yscale("log")
    # ax.set_ylim(1e6, 1.7e7)

    comb_data = {
        "Strategy": ["AR", name_map["Adaptive_FFT_10"], "App-aware"],
        "RUM": [],
    }
    comb_data["RUM"].append(df[(df["Forecaster"] == "AR")]["RUM"].sum())
    comb_data["RUM"].append(
        df[(df["Forecaster"] == name_map["Adaptive_FFT_10"])]["RUM"].sum()
    )
    comb_data["RUM"].append(1.203671e07 + 5.355242e06 + 1.195713e06)

    comb_data_df = pd.DataFrame(comb_data)

    axs[0].set_xlabel("App Class (Based on Invocation Count)")
    axs[0].set_xticklabels(["<1M", "1M-100M", ">100M"])

    sns.barplot(
        x="Strategy",
        y="RUM",
        data=comb_data_df,
        ax=axs[1],
        width=0.8,
        palette="colorblind",
    )
    axs[1].set_ylabel("")
    # set axs[1] labels as X and Y
    axs[1].set_xlabel("")
    # for tick in axs[1].get_xticklabels():
    #     tick.set_rotation(22)
    axs[1].set_xticklabels(["AR Only", "FFT Only", "[FFT,AR,AR]"], rotation=22)
    axs[1].set_title("All Apps", fontsize=10)

    print(comb_data_df)

    plt.tight_layout()
    plt.savefig(save_path + "sec_4_diff_size_apps_v2.pdf")

    plt.close()
