import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import gen_result_df
from metric_calc import objective_function
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "mae") + "/"
save_path = str(Path(__file__).parents[1] / "output_plots") + "/"

mae_save_path = data_dir + "{}_maes.pickle"

MAE_results = {}
RUM_results = {}


def plot_metric(forecasters, result_df):
    args = dict()
    args["x_label"] = "Error"
    args["y_label"] = "Fraction of Applications"
    args["log"] = True
    args["forecaster_num"] = 0

    for forecaster in forecasters:
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

        mae_df = pd.read_pickle(mae_save_path.format(forecaster))
        result_df = result_df.merge(mae_df, on="HashApp")
        print(result_df)

        vals = list(result_df.Metric.to_list())
        args["label"] = "{}-RUM".format(forecaster)
        RUM_results[forecaster] = vals
        # plot(vals, args=args)

        mae_vals = list(np.array(result_df.MAE.to_list()).flatten())

        args["label"] = "{}-MAE".format(forecaster)
        MAE_results[forecaster] = mae_vals
        # plot(mae_vals, args=args)

        result_df.drop("MAE", axis=1, inplace=True)


# def plot(data, args):
#     data = [round(val, 6) for val in data]
#     cdfx = np.sort(data)
#     cdfy = np.linspace(1 / len(data), 1.0, len(data))

#     plt.set_cmap("cividis")
#     linestyle = "-" if args["forecaster_num"] % 2 == 0 else "--"

#     plt.plot(cdfx, cdfy, label=args["label"], linestyle=linestyle)
#     if "log" in args and args["log"] == True:
#         plt.xscale("log")

#     plt.xlabel(args["x_label"].title())
#     plt.ylabel(args["y_label"])
#     # plt.legend()
#     plt.tight_layout()


if __name__ == "__main__":
    forecasters = ["FFT_10", "AR_10"]

    block_size = 504
    percentage = 100
    data_split = "train"
    data_desc = "{}_{}_percent_{}".format(block_size, percentage, data_split)

    result_df = gen_result_df(forecasters, data_desc)
    plot_metric(forecasters, result_df)

    axs = plt.figure(figsize=(6, 2.75), dpi=300, constrained_layout=True).subplots(1, 2)

    fft_mae_gain = [
        "FFT"
        for i in range(len(MAE_results["FFT_10"]))
        if MAE_results["AR_10"][i] > MAE_results["FFT_10"][i]
    ]
    fft_rum_gain = [
        "FFT"
        for i in range(len(RUM_results["FFT_10"]))
        if RUM_results["AR_10"][i] > RUM_results["FFT_10"][i]
    ]

    axs[0].pie(
        [len(fft_mae_gain), len(MAE_results["FFT_10"]) - len(fft_mae_gain)],
        labels=["FFT", "AR_10"],
        autopct="%1.1f%%",
        colors=["#1bcf1b", "#ff7f0e"],
        textprops={"fontsize": 15},
        wedgeprops={"edgecolor": "black"},
    )
    axs[1].pie(
        [len(fft_rum_gain), len(RUM_results["FFT_10"]) - len(fft_rum_gain)],
        labels=["FFT", "AR_10"],
        autopct="%1.1f%%",
        colors=["#1bcf1b", "#ff7f0e"],
        textprops={"fontsize": 15},
        wedgeprops={"edgecolor": "black"},
    )

    axs[0].set_title("MAE", pad=-10, fontsize=15)
    axs[1].set_title("RUM", pad=-10, fontsize=15)

    # # log y axis
    # axs[0].set_yscale("log")
    # axs[1].set_yscale("log")
    # plt.tight_layout()
    plt.savefig(save_path + "sec_4_mae_vs_rum.pdf")
    plt.close()