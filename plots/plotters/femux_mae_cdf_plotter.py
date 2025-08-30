import gc
import pandas as pd
import numpy as np
from time import time
from plotter import plot
from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/azure/"

forecaster_data_path = data_dir + "forecaster_data/concurrency/conc_small_forecast_AR_00.pckl"
stats_save_path = data_dir + "plotter_data/test_mae_cdf_data.pckl"
mae_save_path = data_dir + "results/maes/_maes.pckl"

NUM_SMALL_CHUNKS = 40
NUM_MED_CHUNKS = 4
NUM_LARGE_CHUNKS = 1
NUM_CHUNKS = 4

def gather_test_maes(forecasters, stats, file_description, append=False): 
    mae_df = pd.DataFrame(columns=stats, index=forecasters)
    save_path = stats_save_path.replace("data.", "data_{}.".format(file_description))
    if append:
        mae_df = pd.read_pickle(save_path)

    df = get_femux_df("femux", file_description)

    hashapps = df.HashApp.values.tolist()
    print(len(hashapps))

    del df
    gc.collect()

    for forecaster in forecasters:
        print("getting stats for {}".format(forecaster))
        if "femux" in forecaster:
            df = get_femux_df(forecaster, file_description)
            mae_lists = df.MAE.values.tolist()
            del df
        elif "femux" not in forecaster:
            mae_lists = get_forecaster_maes(forecaster, hashapps)

        ninety_nine_percentile_vals = []
        median_vals = []
        avg_vals = []

        for mae_list in mae_lists:
            if len(mae_list) > 0:
                ninety_nine_percentile_vals.append(np.percentile(mae_list[400:], 99))
                median_vals.append(np.percentile(mae_list[400:], 50))
                avg_vals.append(np.average(mae_list[400:]))

        mae_df.at[forecaster, "Median"] = median_vals
        mae_df.at[forecaster,"99thPercentile"] = ninety_nine_percentile_vals
        mae_df.at[forecaster, "Avg"] = avg_vals

        gc.collect()


    mae_df.to_pickle(save_path)
    print(mae_df)


def get_forecaster_maes(forecaster, hashapps):
    mae_lists = []
    for size in ["small", "medium", "large"]:
        data_path, num_chunks = set_forecaster_mode(forecaster_data_path, size)
        
        forecaster_path = data_path.replace("AR", forecaster)
        
        for chunk_index in range(num_chunks):
            forecaster_df = pd.read_pickle(forecaster_path.replace("_00", "_{:02d}".format(chunk_index)))
            forecaster_df = forecaster_df[forecaster_df["HashApp"].isin(hashapps)]
            mae_lists.extend(forecaster_df.MAE.values.tolist())
        
    return mae_lists


def set_forecaster_mode(data_path, size):
    data_path = data_path.replace("small", size)

    if size == "small":
        num_chunks = NUM_SMALL_CHUNKS
    if size == "medium":
        num_chunks = NUM_MED_CHUNKS
    elif size == "large":
        num_chunks = NUM_LARGE_CHUNKS
    
    return data_path, num_chunks


def plot_mae_cdfs(forecasters, stat, mode, timestep, file_description, extra_description=None):
    cdf_args = dict()

    mae_df = pd.read_pickle(stats_save_path.replace("data.", "data_{}.".format(file_description)))
    
    cdf_args["MAE_Stat"] = stat
    cdf_args["FilterMode"] = "test"
    cdf_args["Mode"] = mode
    cdf_args["Timestep"] = timestep
    cdf_args["Extra"] = extra_description
    cdf_args["Data"] = "azure"
    
    close = False
    for i, forecaster in enumerate(forecasters):
        if i == len(forecasters) - 1:
            close = True

        cdf_args["Forecaster"] = forecaster
        vals = mae_df.loc[forecaster][stat]

        plot(vals, 99, cdf_args, close)

def get_femux_df(forecaster, file_description):
    data_path = mae_save_path.replace("_maes", "{}_maes".format(forecaster))
    data_path = data_path.replace("_maes", "_maes_{}".format(file_description))
    
    dfs = []
    for chunk_index in range(NUM_CHUNKS):
        dfs.append(pd.read_pickle(data_path.replace(".pckl", "_{}.pckl".format(chunk_index))))

    df = pd.concat(dfs, ignore_index=True)
    
    return df

if __name__ == '__main__':
    file_description = "504"
    stats = ["Median", "99thPercentile", "Avg"]
    forecasters = ["IceBreaker", "Holt", "SETAR", "FFT", "AR", "MarkovChain"]

    #gather_test_maes(forecasters , stats, file_description, append=False)
    for stat in stats:
        plot_mae_cdfs(forecasters, stat, "concurrency", 1, file_description)