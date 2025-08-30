import pickle
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import chain, combinations

sys.path.append("../../")
from plots.plotters.plotter import plot

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"

cs_wm_forecaster_path = data_dir + "{}_cold_starts_wasted_mem.pickle"
exec_path = data_dir + "app_exec_time_data.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

plt.rcParams["pdf.fonttype"] = 42
axs = plt.figure(figsize=(6.8, 2.8), dpi=300, constrained_layout=True).subplots(1, 2)

def gen_result_df(forecasters):
    result_df = pd.DataFrame()

    for forecaster in forecasters:
        print(forecaster)
        
        data_path = cs_wm_forecaster_path.format(forecaster)
        forecaster_df = pd.read_pickle(data_path)
        forecaster_df = forecaster_df[['HashApp', 'NumColdStarts', 'NumInvocations', 'MemAllocated', 'MemoryUsed']]

        forecaster_df["SumColdStarts"] = forecaster_df.NumColdStarts.apply(lambda x: sum(x))
        forecaster_df = forecaster_df[forecaster_df.SumColdStarts > 0]
        forecaster_df.drop(["SumColdStarts"], inplace=True, axis=1)

        print(forecaster_df)

        result_df = update_result_df(result_df, forecaster_df, forecaster)
        
    result_df["SkipBlocks"] = result_df.MemAllocated.apply(lambda x : gen_skip_blocks(forecasters, x))

    return result_df


def gen_skip_blocks(forecasters, mem_allocated):
    """Determine which blocks should be skipped due to no traffic (both actual
    and forecaster)

    forecasters: list[str]

    mem_allocated: list[float]
    Amount of memory allocated per block in GBs

    returns: list[bool]
    True if block should be skipped
    """
    
    skip_blocks = []

    for block_index in range(33):#len(mem_allocated[forecasters[0]]) - 1):
        sum_mem_used = 0

        for forecaster in forecasters:
            sum_mem_used += mem_allocated[forecaster][block_index]

        skip_block = True if sum_mem_used < 0.0000001 else False
        skip_blocks.append(skip_block)

    return skip_blocks


def update_result_df(result_df, forecast_df, forecaster):
    """For each application, aggregate all forecaster data into a dict for each app"""

    if result_df.empty:
        result_df = forecast_df
        
        for col in result_df.columns:
            if col != "HashApp":
                result_df[col] = result_df[col].apply(lambda x : {forecaster: x})
        
        return result_df

    result_df = result_df.merge(forecast_df, on="HashApp", how="left", suffixes=(None, "_y"))
    dropped_df = result_df.dropna()

    if len(dropped_df) < len(result_df):
        print("Dropped {} apps due to missing values in {}".format(len(result_df) - len(dropped_df), forecaster))
        result_df = dropped_df

    
    for col in result_df.columns:
        if col == "HashApp" or col.endswith("_y"):
            continue
        
        result_df[col] = result_df.apply(lambda x : update_dict(x[col], x[col + "_y"], forecaster), axis=1) 

        del result_df[col + "_y"]

    return result_df


def update_dict(result_dict, new_vals, forecaster):
    result_dict[forecaster] = new_vals
    return result_dict

def calc_objective_function(forecaster, block_index, num_cold_starts, mem_used, mem_alloc, exec_duration, exec_mode):

    cold_start_time = num_cold_starts[forecaster][block_index] * COLD_START_DURATION
    wasted_memory = mem_alloc[forecaster][block_index] - mem_used[forecaster][block_index]

    if exec_mode:
        exec_in_seconds = max(exec_duration, 1) / 1000
        obj_func = np.sqrt(cold_start_time / exec_in_seconds) + WASTED_MEMORY_WEIGHT * wasted_memory
    else:
        obj_func = COLD_START_TIME_WEIGHT * cold_start_time + WASTED_MEMORY_WEIGHT * wasted_memory

    return obj_func


def objective_function(forecaster, num_cold_starts, mem_used, mem_allocated, skip_blocks, app_level, exec_durations, exec_mode=False):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []
    for block_index in range(33):#len(mem_allocated[forecaster]) - 1):
        if skip_blocks[block_index]:
            continue

        exec_duration = exec_durations[block_index * 504 // 1440] 

        obj_vals.append(calc_objective_function(forecaster, block_index, num_cold_starts, 
                                                mem_used, mem_allocated, exec_duration, exec_mode))

    return sum(obj_vals) if app_level else obj_vals


def plot_bar(forecasters, exec_mode, result_df):
    exec_df = pd.read_pickle(exec_path)
    result_df = result_df.merge(exec_df, on="HashApp", how="left")

    ax = axs[1] if exec_mode else axs[0]
    ylabel = "Exec Time RUM" if exec_mode else "Default RUM"
    rums = []
    names = []

    for forecast_num, forecaster in enumerate(forecasters):        
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["Metric"] = result_df.apply(lambda x : objective_function(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                               x.MemAllocated, x.SkipBlocks, True, x.ExecDurations, exec_mode), axis=1)

        rums.append(result_df.Metric.sum())

        if "exec" in forecaster:
            name = "FeMux-Exec"
        else:
            name = "FeMux"

        names.append(name)

    colours = ["Orange", "Grey"]
    print(rums)
    ax.bar(names, rums, color=colours)
    ax.set_ylabel(ylabel)

if __name__ == '__main__':
    forecasters = []

    forecasters = ["default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics", "default_exec_kmeans_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics_ExecutionTime"]#, "MarkovChain_v3", "FFT_10", "Holt", "ExpSmoothing", "AR", "SETAR", "5_min_keepalive", "10_min_keepalive"]

    block_size = 504
    percentage = 100

    sizes = ["all"]


    result_df = gen_result_df(forecasters)

    for exec_mode in [True, False]:
        plot_bar(forecasters, exec_mode, result_df)

    plt.savefig("../output_plots/sec_6_RUM_comparison.pdf")