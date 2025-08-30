import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["pdf.fonttype"] = 42
axs = plt.figure(figsize=(6.8, 2.8), dpi=300, constrained_layout=True).subplots(1, 2)
plt.fontsize = 14
data_dir = str(Path(__file__).resolve().parents[2] / "data") + "/"

cluster_data = data_dir + "testing_new_kmeans_default_StandardScaler_504_100_percent_df_Density_Linearity_Stationarity_Harmonics.pickle"


def plot_switches_and_forecasts_per_app():
    df = pd.read_pickle(cluster_data)
    df = df.groupby("HashApp").agg({"Cluster_Forecaster": list, "BlockIndex": list}).reset_index()
    df["NumForecasters"] = df.Cluster_Forecaster.apply(lambda x : len(set(x)))
    df["NumSwitches"] = df.Cluster_Forecaster.apply(lambda x: sum([1 for i in range(1, len(x)) if x[i] != x[i - 1]]))

    plot_line(df.NumForecasters.tolist(), "Number of Forecasters")
    plot_line((np.array(df.NumSwitches.tolist()) / 32) * 100, "Block Switches (%)")


def plot_line(data, x_label):
    
    ax = axs[0] if x_label == "Number of Forecasters" else axs[1]
        
    data = [round(val, 6) for val in data]
    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))
    # plt.set_cmap("cividis")
    ax.plot(cdfx, cdfy)
        
    ax.set_xlabel(x_label)
    if x_label == "Number of Forecasters":
        ax.set_ylabel("Fraction of Applications")
    # add grid
    ax.grid(alpha=0.2)
    ax.grid(alpha=0.2)

if __name__ == "__main__":
    plot_switches_and_forecasts_per_app()
    plt.savefig("../output_plots/sec_6_switching_stats.pdf")
