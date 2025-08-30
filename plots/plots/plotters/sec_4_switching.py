import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

output_plots_dir = str(Path(__file__).parents[1] / "output_plots") + "/"
data_dir = str(Path(__file__).parents[2] / "data") + "/"

FORECAST_WINDOW_OFFSET = 120
BLOCK_SIZE = 504
MIN_CONC = 1.6e-5

plt.rc('font', size=12)
os.makedirs(output_plots_dir + "switching_analysis/", exist_ok=True)    


def plot_hashapp(hashapp):
    orig_df = pd.read_pickle(data_dir + "block_switches.pickle")
    forecast_df = pd.read_pickle(data_dir + "block_switches_forecasts.pickle")

    df = orig_df[orig_df.HashApp == hashapp]
    forecasters = df.Cluster_Forecaster.tolist()
    forecasters = [forecaster.removesuffix("_Metrics") for forecaster in forecasters]
    forecasters = list(set(forecasters))
    forecasters = ["5_min_keepalive", "MarkovChain_v3"]
    block_indices = df.BlockIndex.tolist()

    traffic = df.TransformedValues.tolist()[0]
    traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]

    # create a figure and share x-axis with 3 subplots on top of each other
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 3.5), dpi=300, sharex=True)
    fig.subplots_adjust(hspace=-1)

    plt.subplot(3,1,1)

    block_indices = df.BlockIndex.tolist()
    traffic = df.TransformedValues.tolist()[0]
    traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]
    traffic = [np.ceil(t) if t > MIN_CONC else 0 for t in traffic]
    
    plt.vlines(x=[140], ymin=0, ymax=2, colors='red', ls='--', lw=1)

    plt.plot(list(range(len(traffic[550:850]))), traffic[550:850], color = "black", lw=1)

    plt.yticks([0,1,2])
    plt.ylabel("Concurrency")
    plt.tight_layout()
    plt.savefig(output_plots_dir + "sec_4_switching.pdf")

    for i, forecaster in enumerate(["MarkovChain_v3", "5_min_keepalive"]):
        plt.subplot(3,1,i+2)
        cur_df = forecast_df[forecast_df.HashApp == hashapp]
        rum = get_container_err(cur_df, df, forecaster, block_indices)

        plt.vlines(x=[140], ymin=-2, ymax=6, colors='red', ls='--', lw=1)

        color = "blue" if "Markov" in forecaster else "#ba741e"
        label = "Markov Chain" if "Markov" in forecaster else "5-min Keep-alive"
        plt.plot(list(range(len(rum[550:850]))), rum[550:850], label=label, zorder=1, lw=1, color=color)

        plt.legend(prop={'size': 12})
        plt.ylabel("Error")
        if i == 1:
            plt.xlabel("Time (m)")
        plt.tight_layout()
        plt.savefig(output_plots_dir + "sec_4_switching.pdf".format(hashapp[:5]))
    
    plt.close()


def get_container_err(forecast_df, df, forecaster, block_indices):
    df = df.merge(forecast_df[["HashApp", forecaster]], on="HashApp", how="right")
    df.rename(columns={forecaster: "ForecastedValues"}, inplace=True)

    traffic = df.TransformedValues.tolist()[0]
    traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]
    traffic = [np.ceil(t) if t > MIN_CONC else 0 for t in traffic]
    
    forecasts = forecast_df[forecaster].tolist()[0]
    forecasts = forecasts[block_indices[0] * BLOCK_SIZE - 120: (block_indices[-1] + 1) * BLOCK_SIZE - 120]
    forecasts = forecasts.flatten()
    forecasts = [np.ceil(f) if f > MIN_CONC else 0 for f in forecasts]

    err = [forecasts[i] - traffic[i] for i in range(len(traffic))]

    return err


if __name__ == "__main__":
    plot_hashapp("b6e0ad5c7b1250021ac0acdb7651ef2885e092f8cbce13946f8a973e5da427be")
