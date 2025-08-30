import pandas as pd
import numpy as np
import pickle
import sys
import gc
import os
from time import time
from os.path import exists
from time import strftime
from pathlib import Path

data_dir = str(Path(__file__).resolve().parents[2] / "data") + "/azure/"

transformed_data_path = data_dir + "transformed_data/concurrency/small_app_conc_00.pickle"
memory_data_path = data_dir + "preproc_data/memory_data.pickle"
exec_time_path = data_dir + "preproc_data/app_exec_time_data.pickle"
forecaster_data_path = data_dir + "forecaster_data/concurrency/conc_small_forecast_AR_00.pickle"
cached_data = data_dir + "train_test_split/cached_app_data_{}_{}.pickle"
clustering_data_path = data_dir + "clustering/post_processed_data/"
corrupt_hashapp_path = data_dir + "preproc_data/corrupt_hashapps.pickle"

training_hashapps = data_dir + "train_test_split/{}_training_apps.pickle"
test_hashapps = data_dir + "train_test_split/{}_test_apps.pickle"
workload_data = data_dir + "knative_deployment_data/workload_invocation_data_max_63_100_apps.pickle"

NUM_SMALL_CHUNKS = 40
NUM_MED_CHUNKS = 4
NUM_LARGE_CHUNKS = 1
STATIC_KEEPALIVE_WINDOW = 10

MINUTES_PER_DAY = 1440
NUM_DAYS = 12

def add_exec_times(df):
    if "ExecDurations" in df.columns:
        return df
    
    exec_df = pd.read_pickle(exec_time_path)

    df = df.merge(exec_df, on="HashApp", how="left")
    
    return df


def add_transform_values(df):
    """Add values from transformation dataframe
    cols: list[str]
    columns to include:
    TransformedValues: transformed trace values
    ContainerInvocationsPerMin: number of container invocations per minute
    """
    target_hashapps = df.HashApp.tolist()

    transform_dfs = []

    for size in ["small", "medium", "large"]:
        data_path, num_chunks = set_data_mode(transformed_data_path, size)

        for chunk_index in range(num_chunks):
            transform_df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(chunk_index)))
            transform_df = transform_df[["HashApp", "TransformedValues", "ContainerInvocationsPerMin"]]

            transform_df = transform_df[transform_df.HashApp.isin(target_hashapps)]
            transform_dfs.append(transform_df)
            gc.collect()

    transform_df = pd.concat(transform_dfs, ignore_index=True)

    df = df.merge(transform_df, on="HashApp", how="left")
    
    return df


def set_forecaster(df, forecaster, femux_path):
    if forecaster == "Oracle":
        return df

    target_hashapps = df.HashApp.tolist()

    forecaster_dfs = [] 

    if "femux" in forecaster:
        print(femux_path)
        return df.merge(pd.read_pickle(clustering_data_path + femux_path))

    for size in ["small", "medium", "large"]:
        data_path, num_chunks = set_data_mode(forecaster_data_path, size)
        
        forecaster_path = data_path.replace("AR", forecaster)
        
        for chunk_index in range(num_chunks):
            try:
                forecaster_df = pd.read_pickle(forecaster_path.replace("00", "{:02d}".format(chunk_index)))
            except Exception as e:
                continue

            forecaster_df = forecaster_df[forecaster_df.HashApp.isin(target_hashapps)]
            forecaster_df = forecaster_df[["HashApp", "ForecastedValues"]]
            forecaster_dfs.append(forecaster_df)

    forecaster_df = pd.concat(forecaster_dfs, ignore_index=True)

    df = df.merge(forecaster_df, on="HashApp", how="left")

    return df


def set_data_mode(data_path, size):
    data_path = data_path.replace("small", size)

    if size == "small":
        num_chunks = NUM_SMALL_CHUNKS
    if size == "medium":
        num_chunks = NUM_MED_CHUNKS
    elif size == "large":
        num_chunks = NUM_LARGE_CHUNKS
    
    return data_path, num_chunks


def add_mem_values(df):
    mem_df = pd.read_pickle(memory_data_path)

    df = df.merge(mem_df, on="HashApp", how="left")

    return df


def gen_static_keepalive(trace, keepalive_window=STATIC_KEEPALIVE_WINDOW):
    forecasts = np.empty((len(trace), 1))

    # forecast for first timestep is 0
    forecasts[0][0] = 0

    for index in range(1, len(trace)):
        start_index = max(index - keepalive_window, 0)
        forecasts[index][0] = max(trace[start_index:index])

    return forecasts


def gather_data_for_results(mode, data_percentage=0):
    cache_path = cached_data.format(data_percentage, mode)
    
    if exists(cache_path):
        print("using cached data", strftime("%H:%M:%S"))
        df = pd.read_pickle(cache_path)

    else:
        df = init_df(mode, data_percentage)

        print("adding transform values", strftime("%H:%M:%S"))
        df = add_transform_values(df)
        
        print("adding memory values", strftime("%H:%M:%S"))
        df = add_mem_values(df)

        print("adding exec times", strftime("%H:%M:%S"))
        df = add_exec_times(df)

        print("caching data to {}".format(cache_path))
        df = df[["HashApp", "TransformedValues", "AverageMemUsage", "ContainerInvocationsPerMin", "ExecDurations"]]
        df.to_pickle(cache_path)
    
    # some apps don't have memory usage recorded, so we remove them
    df.drop_duplicates(subset=["HashApp"], inplace=True)
    gc.collect()

    return df


def init_df(mode, data_percentage):
    if mode == "deployment":
        return pd.read_pickle(workload_data)[["HashApp"]]

    """Create dataframe with a HashApp column from representative sample of hashapps

    mode: str
    "train" for training data, or "test" for test data

    data_percentage: int
    what percent of the data to use (e.g., 80)
    """
    hashapp_list = []
    save_path = training_hashapps if mode == "train" else test_hashapps

    for size in ["small", "medium", "large"]:

        with open(save_path.format(size), 'rb') as f:
            hashapps = pickle.load(f)

        num_apps = int(np.ceil(len(hashapps) * data_percentage / 100))

        if mode == "train": 
            hashapp_list.extend(hashapps[:num_apps]) 
        if mode == "test":
            hashapp_list.extend(hashapps[-num_apps:]) 
    
    return pd.DataFrame({"HashApp": hashapp_list}) 


def clean_data(df, block_size):
    num_elements = (NUM_DAYS * MINUTES_PER_DAY // block_size) * block_size
    
    # some apps don't have memory values
    df = df.dropna()
    df.drop(df[df["HashApp"] == "cb34fd874e255ddeaf1a38d86e7f3a41dffb0efdee8e46329d4ba8c2dad0fab5"].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # some apps are missing execution or invocation data
    with open(corrupt_hashapp_path, 'rb') as f:
        corrupt_hashapps = pickle.load(f)

    df = df[~df.HashApp.isin(corrupt_hashapps)]

    # some apps don't have any invocations in the first 12 days    
    df["Check"] = df.TransformedValues.apply(lambda x : sum(x[:num_elements]))
    df = df[df.Check > 0]
    df.drop(["Check"], inplace=True, axis=1)

    return df