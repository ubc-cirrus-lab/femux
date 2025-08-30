import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("../../code")
from results.utils import add_mem_values, add_transform_values, set_forecaster
from results.gen_results import gen_multiproc_results
from clustering.utils import set_metric

output_plots_dir = str(Path(__file__).parents[1] / "output_plots" / "azure") + "/"
data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"

cluster_file = data_dir + "clustering/testing_output/testing_kmeans_default_markov_v3_StandardScaler_100_percent_df.pickle"
traffic_path = data_dir + "transformed_data/concurrency/app/{}_app_conc_{:02d}.pickle"
memory_data_path = data_dir + "preproc_data/memory_data.pickle"
save_path = output_plots_dir + "switching_analysis/"

FORECAST_WINDOW_OFFSET = 120
BLOCK_SIZE = 504
MIN_CONC = 1.6e-5

plt.rc('font', size=12)
os.makedirs(output_plots_dir + "switching_analysis/", exist_ok=True)    

def get_block_switches():
    df = pd.read_pickle(cluster_file)[0]

    df = df.reset_index(drop=True)

    chosen_names = []
    block_indices = []
    hashapps = []
    indices_to_keep = []

    # Plot the switching analysis
    for i, row in df.iterrows():
        chosen_names.append(row.Oracle_Forecaster)
        block_indices.append(row.BlockIndex)
        hashapps.append(row.HashApp)

        if i > 1:
            chosen_names.pop(0)
            block_indices.pop(0)
            hashapps.pop(0)

        if i > 0:            
            if chosen_names[0] != chosen_names[1] and block_indices[0] == block_indices[1] - 1 and hashapps[0] == hashapps[1]:
                rum_diff = abs(row[chosen_names[0]] - row[chosen_names[1]])
                rum_diff_proportion =  rum_diff / (row[chosen_names[0]] + 1e-6)
                
                if rum_diff_proportion > 0.3 and rum_diff > 10:
                    indices_to_keep.extend([i-1, i])

    indices_to_keep = sorted(list(set(indices_to_keep)))

    df = df.iloc[indices_to_keep]
    df = df.reset_index(drop=True)

    df = add_transform_values(df)
    
    print(df[["HashApp", "BlockIndex", "Oracle_Forecaster"]])
    
    df.to_pickle(save_path + "block_switches.pickle")


def get_forecasts():
    df = pd.read_pickle(save_path + "block_switches.pickle")
    hashapps = df.HashApp.tolist()

    selected_apps = []
    for app in ["b5e73", "b6e0a", "b386e", "bad6b", "bc668", "c8e84", "c2336", "caa6a", "cd45a", "d0fe5", "d3c3f", "d260d", "d638d", "d472b", "d8262", "d9946", "dac62"]:
        selected_apps.extend([hashapp for hashapp in hashapps if hashapp.startswith(app)])

    selected_apps = list(set(selected_apps))

    df = df[df.HashApp.isin(selected_apps)]

    df.drop_duplicates(subset=["HashApp"], inplace=True)

    for forecaster in ["AR", "SETAR", "MarkovChain_v3", "5_min_keepalive", "10_min_keepalive", "Holt", "ExpSmoothing"]:
        df = set_forecaster(df, forecaster, "fill")
        df.rename(columns={"ForecastedValues": forecaster}, inplace=True)

    print("done getting forecasts")

    df.to_pickle(save_path + "block_switches_forecasts.pickle")



def plot_traffic():
    switch_df = pd.read_pickle(save_path + "block_switches.pickle")

    for hashapp in switch_df.HashApp.tolist():
        df = switch_df[switch_df.HashApp == hashapp]
        block_indices = df.BlockIndex.tolist()
        traffic = df.TransformedValues.tolist()[0]
        traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]

        plt.plot(list(range(len(traffic))), traffic, label="traffic", zorder=1, lw=0.5)

        plt.legend()
        plt.xlabel("Time (m)")
        plt.ylabel("Average Concurrency")
        plt.tight_layout()
        plt.savefig(save_path + "3_blocks_traffic_{}.pdf".format(hashapp[:5]))
        plt.close()


def get_error(forecast_df, df, forecaster, block_indices):
    df = df.merge(forecast_df[["HashApp", forecaster]], on="HashApp", how="right")
    df.rename(columns={forecaster: "ForecastedValues"}, inplace=True)

    traffic = df.TransformedValues.tolist()[0]
    traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]
    
    forecasts = forecast_df[forecaster].tolist()[0]
    forecasts = forecasts[block_indices[0] * BLOCK_SIZE - 120: (block_indices[-1] + 1) * BLOCK_SIZE - 120]

    err = [traffic[i] - forecasts[i] for i in range(len(traffic))]

    return err


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


def get_RUM(forecast_df, df, forecaster, block_indices):
    df = df.merge(forecast_df[["HashApp", forecaster]], on="HashApp", how="right")
    df.rename(columns={forecaster: "ForecastedValues"}, inplace=True)

    df = add_mem_values(df)
    
    df = gen_multiproc_results(df, forecaster, forecast_window=120, forecast_len=1, block_size=1)

    df = set_metric(df, "default", 1)
    
    rum = df.Metric.tolist()[0]
    return rum[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]


def plot_traffic_err_rum():
    orig_df = pd.read_pickle(save_path + "block_switches.pickle")
    forecast_df = pd.read_pickle(save_path + "block_switches_forecasts.pickle")
    print(forecast_df.HashApp.tolist())

    for hashapp in forecast_df.HashApp.tolist():
        df = orig_df[orig_df.HashApp == hashapp]
        forecasters = df.Cluster_Forecaster.tolist()
        forecasters = [forecaster.removesuffix("_Metrics") for forecaster in forecasters]
        forecasters = list(set(forecasters))
        block_indices = df.BlockIndex.tolist()

        block_indices = df.BlockIndex.tolist()
        traffic = df.TransformedValues.tolist()[0]
        traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]
        
        plt.subplot(2,1,1)
        for forecaster in forecasters:
            cur_df = forecast_df[forecast_df.HashApp == hashapp]
            rum = get_RUM(cur_df, df, forecaster, block_indices)

            #plt.vlines(x=[504 * (i+1) for i in range(len(block_indices))], ymin=0, ymax=max(rum), colors='red', ls='--', lw=1)

            plt.plot(list(range(len(rum))), rum, label=forecaster, zorder=1, lw=1, alpha = 0.8)

            plt.legend(prop={'size': 9})
            plt.ylabel("RUM")
            plt.tight_layout()
            plt.savefig(save_path + "3_blocks_combo_{}.pdf".format(hashapp[:5]))

        plt.subplot(2,1,2)

        block_indices = df.BlockIndex.tolist()
        traffic = df.TransformedValues.tolist()[0]
        traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]

        plt.plot(list(range(len(traffic))), traffic, color = "black", lw=1)

        plt.xlabel("Time (m)")
        plt.ylabel("# Containers")
        plt.tight_layout()
        plt.savefig(save_path + "3_blocks_combo_{}.pdf".format(hashapp[:5]))
        plt.close()


def plot_hashapp(hashapp):
    orig_df = pd.read_pickle(save_path + "block_switches.pickle")
    forecast_df = pd.read_pickle(save_path + "block_switches_forecasts.pickle")

    df = orig_df[orig_df.HashApp == hashapp]
    forecasters = df.Cluster_Forecaster.tolist()
    forecasters = [forecaster.removesuffix("_Metrics") for forecaster in forecasters]
    forecasters = list(set(forecasters))
    forecasters = ["5_min_keepalive", "MarkovChain_v3"]
    block_indices = df.BlockIndex.tolist()

    traffic = df.TransformedValues.tolist()[0]
    traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]

    plt.subplot(3,1,1)

    block_indices = df.BlockIndex.tolist()
    traffic = df.TransformedValues.tolist()[0]
    traffic = traffic[block_indices[0] * BLOCK_SIZE: (block_indices[-1] + 1) * BLOCK_SIZE]
    traffic = [np.ceil(t) if t > MIN_CONC else 0 for t in traffic]
    
    #plt.vlines(x=[140], ymin=0, ymax=2, colors='red', ls='--', lw=1)

    plt.plot(list(range(len(traffic[550:850]))), traffic[550:850], color = "black", lw=1)

    plt.yticks([0,1,2])
    plt.ylabel("Concurrency")
    plt.tight_layout()
    plt.savefig(save_path + "3_blocks_line_{}.pdf".format(hashapp[:5]))

    for i, forecaster in enumerate(["MarkovChain_v3", "5_min_keepalive"]):
        plt.subplot(3,1,i+2)
        cur_df = forecast_df[forecast_df.HashApp == hashapp]
        rum = get_container_err(cur_df, df, forecaster, block_indices)

        #plt.vlines(x=[140], ymin=-2, ymax=6, colors='red', ls='--', lw=1)

        color = "blue" if "Markov" in forecaster else "orange"
        label = "Markov Chain" if "Markov" in forecaster else "5-min Keep-alive"
        plt.plot(list(range(len(rum[550:850]))), rum[550:850], label=label, zorder=1, lw=1, color=color)

        plt.legend(prop={'size': 12})
        plt.ylabel("Error")
        plt.ylim((-2.5,6.5))
        if i == 1:
            plt.xlabel("Time (m)")
        plt.tight_layout()
        plt.savefig(save_path + "3_blocks_line_{}.pdf".format(hashapp[:5]))
    
    plt.close()


def make_err_dict(forecast_df, df, forecasters, block_indices, hashapp):
    cur_df = forecast_df[forecast_df.HashApp == hashapp]
    first_errs = get_container_err(cur_df, df, forecasters[0], block_indices)
    second_errs = get_container_err(cur_df, df, forecasters[1], block_indices)
    both = [first_errs[i] if first_errs[i] == second_errs[i] else 0 for i in range(len(first_errs))]    
    first_errs = [first_errs[i] if first_errs[i] != second_errs[i] else 0 for i in range(len(first_errs))]
    second_errs = [second_errs[i] if first_errs[i] != second_errs[i] else 0 for i in range(len(second_errs))]
    
    return {forecasters[0]: first_errs, forecasters[1]: second_errs, "Both": both}




if __name__ == "__main__":
    #get_block_switches()
    #plot_traffic()
    #get_forecasts()
    #plot_traffic_err_rum()
    plot_hashapp("b6e0ad5c7b1250021ac0acdb7651ef2885e092f8cbce13946f8a973e5da427be")