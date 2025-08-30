import gc
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

forecaster_data_path = data_dir + "forecaster_data/concurrency/conc_small_forecast_AR_00.pickle"
inv_exec_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"
exec_time_path = data_dir + "preproc_data/app_exec_time_data.pickle"
it_forecast_path = data_dir + "forecaster_data/idletime/idletime_small_forecast_{}_{:02d}.pickle"

NUM_SMALL_FILES = 40
NUM_MED_FILES = 4
NUM_LARGE_FILES = 1
MINUTES_PER_DAY = 1440

COLD_START_DURATION = 0.808
DEFAULT_CS_TIME_WEIGHT = 1
DEFAULT_WASTED_MEM_WEIGHT = 1 / 99.69


def add_exec_times(df):
    exec_df = pd.read_pickle(exec_time_path)

    df = df.merge(exec_df, on="HashApp", how="left")
    
    return df


def add_idletime_forecasts(femux_df, forecaster):
    """Add idle time forecasts for FeMux.

    Some applications won't have idle time forecasts since they have frequent invocations
    """
    print("Adding idle time forecasts for {}".format(forecaster))
    target_hashapps = femux_df.HashApp.tolist()

    forecast_dfs = []
    for filenum in range(NUM_SMALL_FILES):
        forecast_df = pd.read_pickle(it_forecast_path.format(forecaster, filenum))
        forecast_df = forecast_df[["HashApp", "ForecastedValues"]]
        forecast_dfs.append(forecast_df[forecast_df.HashApp.isin(target_hashapps)])

    gc.collect()
    forecast_df = pd.concat(forecast_dfs, ignore_index=True)
    forecast_df.rename(columns={"ForecastedValues": "IdleTimeForecasts"}, inplace=True)
    femux_df = femux_df.merge(forecast_df, on="HashApp", how="left")

    return femux_df


def add_forecaster_values(cluster_df, forecaster):
    if "HashApp" not in cluster_df.columns:
        cluster_df.reset_index(inplace=True)

    target_hashapps = cluster_df.HashApp.tolist()
    
    # label forecasted values that belong to this forecaster
    forecast_col_name = "ForecastedValues_{}".format(forecaster)
    forecast_dfs = []

    for size in ["small", "medium", "large"]:
        print("Adding values for {} {}".format(forecaster, size))
        data_path, num_files = set_forecaster_mode(forecaster_data_path, size)
        
        forecaster_path = data_path.replace("AR", forecaster)
                
        for chunk_index in range(num_files):
            forecast_df = pd.read_pickle(forecaster_path.replace("_00", "_{:02d}".format(chunk_index)))

            forecast_df = forecast_df[["HashApp", "ForecastedValues"]]
            forecast_df.rename(columns={"ForecastedValues": forecast_col_name}, inplace=True)

            forecast_dfs.append(forecast_df[forecast_df.HashApp.isin(target_hashapps)])

        del forecast_df
        gc.collect()
        
    forecast_df = pd.concat(forecast_dfs, ignore_index=True)
    cluster_df = cluster_df.merge(forecast_df, on="HashApp", how="left")
    
    return cluster_df


def set_forecaster_mode(data_path, size):
    data_path = data_path.replace("small", size)

    if size == "small":
        num_files = NUM_SMALL_FILES
    if size == "medium":
        num_files = NUM_MED_FILES
    elif size == "large":
        num_files = NUM_LARGE_FILES
    
    return data_path, num_files


def set_75p_metric(df, num_blocks, block_size, weight_mode):
    agg_cs_pct_list = [[] for i in range(num_blocks)]
    cs_weight = 10 if "10_cs" in weight_mode else 1
    
    invocation_df = pd.read_pickle(inv_exec_path.format(12))
    df = df.merge(invocation_df[["HashApp", "InvocationsPerMin"]], on="HashApp", how="left")

    # Get the cold start percentages across all apps for each block
    df.apply(lambda x : agg_cs_pct_per_block_across_apps(x.NumColdStarts, x.InvocationsPerMin, 
                                                        agg_cs_pct_list, num_blocks, block_size), axis=1)

    # For each block get 75th percentile cold start percentage across all apps
    for i in range(len(agg_cs_pct_list)):
        agg_cs_pct_list[i] = np.nanpercentile(np.array(agg_cs_pct_list[i]), 75)

    df["Metric"] = df.apply(lambda x : gen_75p_metric(x.MemoryUsed, x.MemAllocated, x.BaselineMemWasted, 
                                                        agg_cs_pct_list, num_blocks, cs_weight), axis=1)
    
    return df


def gen_75p_metric(mem_used, mem_alloc, baseline_wasted_mem, agg_cs_pct_list, num_blocks, cs_weight):
    metrics = []
    for block_index in range(num_blocks):
        wasted_mem = mem_alloc[block_index] - mem_used[block_index]
        obj_val = 0

        if baseline_wasted_mem[block_index] > 0:
            obj_val = cs_weight * agg_cs_pct_list[block_index] + (wasted_mem / baseline_wasted_mem[block_index]) * 100
        
        metrics.append(obj_val)

    return metrics


def agg_cs_pct_per_block_across_apps(cs_blocks, inv_list, agg_cs_pct_list, num_blocks, block_size):
    """Each trace is split into n blocks index 0 to n.
    We want to aggregate the cold start percentage of block b_i from 
    all applications into one list, for i in [0,n).
    """
    for block_index in range(num_blocks):
        num_invocations = sum(inv_list[block_index * block_size : (block_index + 1) * block_size])
        cs_pct = (cs_blocks[block_index] / num_invocations) * 100 if num_invocations > 0 else 0
        agg_cs_pct_list[block_index].append(cs_pct) 


def set_metric(df, weight_mode, block_size):
    """Set the weights and forecaster name based on the mode
    weight_mode: str
        "4_cs": 4 times higher cold start weight
        "4_wm": 4 times higher wasted memory weight

    returns: str
    updated forecaster name
    """ 
    cs_time_weight = DEFAULT_CS_TIME_WEIGHT
    wasted_mem_weight = DEFAULT_WASTED_MEM_WEIGHT

    if "4_cs" in weight_mode:
        cs_time_weight *= 4
    elif "4_wm" in weight_mode:
        wasted_mem_weight *= 4

    df = add_exec_times(df)

    invocation_df = pd.read_pickle(inv_exec_path.format(12))
    df = df.merge(invocation_df[["HashApp", "InvocationsPerMin"]], on="HashApp", how="left")

    df["Metric"] = df.apply(lambda x : objective_function(x.NumColdStarts, x.MemAllocated - x.MemoryUsed, 
                                                        x.ExecDurations, x.InvocationsPerMin, weight_mode, 
                                                        block_size, cs_time_weight, wasted_mem_weight), axis=1)

    return df


def objective_function(num_cold_starts, mem_wasted, exec_durations, inv_per_min, weight_mode, block_size,
                        cs_time_weight, wasted_mem_weight):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_wasted: list[float]
    memory wasted per block in GB-s

    exec_durations: list[int]
    execution duration per day in milliseconds

    inv_per_min: list[int]
    Number of invocations per minute at an application level
    
    weight_mode: str
    indicates how to calculate objective value
    """

    obj_vals = []

    for block_index in range(len(mem_wasted)):
        obj_val = 0
        cur_day = block_index * block_size // MINUTES_PER_DAY

        num_invocations = sum(inv_per_min[block_index * block_size : (block_index + 1) * block_size])

        # might occur during a day-crossing, so take larger one
        exec_duration = max(exec_durations[cur_day], exec_durations[cur_day - 1])

        if num_invocations > 0 and mem_wasted[block_index] > 0:
            obj_val = calc_objective_function(num_cold_starts[block_index], mem_wasted[block_index], num_invocations,
                                            weight_mode, exec_duration, cs_time_weight, wasted_mem_weight)

        obj_vals.append(obj_val)

    return obj_vals


def calc_objective_function(num_cold_starts, wasted_mem, num_invocations, weight_mode, exec_duration,
                            cs_time_weight, wasted_mem_weight):
    """ Calculate the objective function. weight_mode indicates when to consider execution duration relative to cold start time, 
    or the objective value relative to number of invocations to stop popular applications from dominating the optimization.
    """

    cold_start_time = num_cold_starts * COLD_START_DURATION

    if "exec" in weight_mode:
        obj_func = cs_time_weight * np.sqrt(cold_start_time / max(exec_duration, 1)) + wasted_mem_weight * wasted_mem
    else:
        obj_func = cs_time_weight * cold_start_time + wasted_mem_weight * wasted_mem

    return obj_func


def gen_wasted_mem_baseline(static_keepalive_df):
    static_keepalive_df["BaselineMemWasted"] = static_keepalive_df.apply(lambda x : x.MemAllocated - x.MemoryUsed, axis=1)
    return static_keepalive_df[["HashApp", "BaselineMemWasted"]]