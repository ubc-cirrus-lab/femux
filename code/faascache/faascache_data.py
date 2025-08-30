import gc
import pickle
import pandas as pd
import numpy as np
import sys
from os import listdir
sys.path.append("../../code/")
from concurrent.futures import ProcessPoolExecutor
import results.utils as utils
import matplotlib.pyplot as plt
from time import strftime
from datetime import datetime
from math import ceil
from pathlib import Path

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"
faascache_data_dir = data_dir + "faascache/"

mem_logs_path = faascache_data_dir + "GD-{}-{}-a-purecachehist.csv"
analyzed_stats_path = faascache_data_dir + "GD-{}-{}-a.pckl"
lambda_data_path = faascache_data_dir + "2523-a.pckl"

preproc_path = data_dir + "preproc_data/invocation_data/preprocessed_data_{:02d}.pickle"
faascache_hashapps = data_dir + "faascache/faascache_hashapps.pickle"
faascache_results = data_dir + "faascache/{}_results.pickle"
azure_data = data_dir + "azure_data/function_durations_percentiles.anon.d01.csv"
inv_exec_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"

BLOCK_SIZE = 504
STATIC_KEEPALIVE_WINDOW = 10

MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
MB_to_GB = 1000
MIN_CONC = 1.6e-05
MIN_FORECAST = MIN_CONC / 2 

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69
SECONDS_IN_DAY = 86400

def gen_femux_forecasts(num_workers, faascache_hashapps, forecaster):
    forecast_window = [120] * num_workers
    forecast_len = [1] * num_workers 
    block_size = 504
    
    df = utils.gather_data_for_results("test", 100)
    df = utils.clean_data(df, block_size)
    df = df[df.HashApp.isin(faascache_hashapps)]
    
    if forecaster == "Oracle":
        pass
    elif "keepalive" not in forecaster:
        df = utils.set_forecaster(df, "femux", "processed_testing_df_{}_markov_v3_StandardScaler_504_100_percent_test_Density_Linearity_Stationarity_Harmonics.pickle".format(forecaster)) 
    elif forecaster:
        df = utils.set_forecaster(df, "10_min_keepalive", "")


    dfs = np.array_split(df, num_workers)
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(gen_multiproc_results, dfs, [forecaster] * num_workers, forecast_window, forecast_len)

    result_df = pd.concat(results, ignore_index=True)

    return result_df

def gen_multiproc_results(df, forecaster, forecast_window, forecast_len):
    if forecaster == "Oracle":
        forecast_window = 0
        df["ForecastedValues"] = df.TransformedValues.apply(lambda x : [[x[i]] for i in range(forecast_window, len(x))])


    print("Starting cold starts", strftime("%H:%M:%S"))
    df[["NumColdStarts", "NumInvocations"]] = df.apply(lambda x : gen_cold_starts(x.ForecastedValues,
                                                                    x.TransformedValues,
                                                                    x.ContainerInvocationsPerMin,
                                                                    forecast_window,
                                                                    forecast_len), axis=1)

    gc.collect()
    print("Starting memory calculations", strftime("%H:%M:%S"))

    df[["MemAllocated", "MemoryUsed"]] = df.apply(lambda x : gen_mem_usage(x.ForecastedValues,
                                                                x.TransformedValues,
                                                                x.AverageMemUsage[0],
                                                                x.ContainerInvocationsPerMin,
                                                                forecast_window,
                                                                forecast_len), axis=1)

    return df


def gen_cold_starts(forecasted_vals, trace, invocations, 
                        forecast_window, forecast_len):
    """ Calculates the number of cold starts for each block

    forecasted_vals: list[float]

    trace: list[float]

    num_invocations: int

    forecast_window: int

    forecast_len: int
    """
    trace = trace[:len(trace) - forecast_len]

    # use static keepalive for timesteps before forecasting starts
    #static_keepalive = utils.gen_static_keepalive(trace[:forecast_window])
    #forecasted_vals = np.concatenate((static_keepalive, forecasted_vals), dtype=object)

    cold_start_sum = 0
    container_invocation_sum = 0
    
    end_minute = 1440

    for cur_minute in range(end_minute - 1):
        num_predicted_containers = 0 if forecasted_vals[cur_minute][0] < MIN_FORECAST else ceil(forecasted_vals[cur_minute][0])
        num_containers = 0 if trace[cur_minute] < MIN_CONC else ceil(trace[cur_minute])
        container_invocation_sum += invocations[cur_minute]

        num_cold_starts = max(0, num_containers - num_predicted_containers)

        # Average concurrency can be positive due to invocations from previous minutes,
        # so we can only count cold starts for invocations that occur in the current minute
        cold_start_sum += min(num_cold_starts, invocations[cur_minute])

    return pd.Series([cold_start_sum, container_invocation_sum])


def gen_mem_usage(forecasted_vals, trace, mem_val, invocations,
                    forecast_window, forecast_len):
    """ Calculates the resource utilization for each block

    forecasted_vals: list[float]

    trace: list[float]

    mem_vals: list[int]
    Memory usage per instance of application in MB

    forecast_window: int

    forecast_len: int
    """
    trace = trace[:len(trace) - forecast_len]

    
    mem_usage = mem_val / MB_to_GB
    #static_keepalive = utils.gen_static_keepalive(trace[:forecast_window])
    
    # use static keepalive for timesteps before forecasting starts
    #forecasted_vals = np.concatenate((static_keepalive, forecasted_vals), dtype=object)
    
    end_minute = 1440

    total_mem_allocated = 0
    total_mem_used = 0

    for cur_minute in range(end_minute - 1):
        cur_trace_val = 0 if trace[cur_minute] < MIN_CONC else trace[cur_minute]
        
        num_predicted_containers = 0 if forecasted_vals[cur_minute][0] < MIN_FORECAST else ceil(forecasted_vals[cur_minute][0])
        num_actual_containers = ceil(cur_trace_val)

        # predicted containers are kept alive throughout the whole minute
        predicted_mem_alloc = num_predicted_containers * mem_usage * SECONDS_PER_MIN

        # when FeMux underpredicts the number of containers, include the memory used during execution for
        # containers that experience cold starts (first n invocations within the minute where n 
        # is the number of cold starts). These are kept alive until the end of the minute.
        remaining_mem_alloc = calc_mem_alloc(num_actual_containers, num_predicted_containers, 
                                        mem_usage, invocations[cur_minute])
        
        mem_allocated = remaining_mem_alloc + predicted_mem_alloc

        # memory utilization only counts the memory used during execution
        mem_used = cur_trace_val * mem_usage * SECONDS_PER_MIN

        # there can be a minute that has no new invocations but is executing existing invocations,
        # in this case, we may only allocate the amount of memory that is in use
        total_mem_allocated += max(mem_allocated, mem_used)
        total_mem_used += mem_used 

    return pd.Series([total_mem_allocated, total_mem_used])


def calc_mem_alloc(num_actual_containers, num_predicted_containers, mem_usage, num_invocations):
    num_invocations = round(num_invocations)
    if round(num_invocations) == 0:
        return 0
    
    num_extra_containers = max(round(num_actual_containers - num_predicted_containers), 0)
    mem_alloc = 0

    iat = SECONDS_PER_MIN / num_invocations
    shift = iat / 2
    cur_second = shift

    # when cold start occurs keep the container alive for remainder of the minute.
    # we only count cold starts based on the number of conatiner invocations that 
    # actually came in during the minute
    for _ in range(min(num_extra_containers, num_invocations)):
        cur_keepalive = SECONDS_PER_MIN - cur_second 

        mem_alloc += mem_usage * cur_keepalive
        cur_second += iat

    return mem_alloc


def gen_faascache_wasted_mem(cache_sizes, num_funcs):
    wasted_mems = []
    
    for cache_size in cache_sizes:

        try:
            # cols => time, used_mem, running_mem, pure_cache
            df = pd.read_csv(mem_logs_path.format(num_funcs, cache_size))
        except:
            print(mem_logs_path.format(num_funcs, cache_size))
            raise

        start = {"time":0, "used_mem":df.at[0, "used_mem"], "running_mem": df.at[0, "running_mem"], "pure_cache":0}
        end = {"time": 24 * 59 * 60 * 1000, "used_mem": df.at[len(df)-1, "used_mem"], "running_mem": df.at[len(df)-1, "running_mem"], "pure_cache":0}
        df2 = pd.DataFrame([start, end], columns=df.columns)
        df = pd.concat([df, df2])

        sort = df.sort_values(by=["time"])
        dedup = sort.drop_duplicates(subset=["time"], keep="last")
        dedup.index = (dedup["time"] / 1000).apply(datetime.fromtimestamp)
        # upsample to second detail since there may be gaps
        # then downsample to minute buckets for
        dedup = dedup.resample("S").mean().interpolate().resample("1Min").interpolate()

        d = dedup["pure_cache"].to_numpy(copy=True)
        if len(d) != 1440:
            d.resize(1440)
    
        sum_wasted_mem = sum(d) / MB_to_GB
        # saved as numpy array in one minute buckets of average memory wasted across the minute,
        # so we multiply by 60 to get GB-s
        wasted_mems.append(sum_wasted_mem * 60)

    return wasted_mems


def gen_faascache_cs(cache_sizes, num_funcs):
    cold_start_list = []
    num_accesses_list = []
    num_dropped_list = []

    for cache_size in cache_sizes:
        analyzed = pd.read_pickle(analyzed_stats_path.format(num_funcs, cache_size))
        num_invocations = analyzed[3]
        analyzed = analyzed[1]

        num_cold_starts = 0
        num_accesses = 0
        for func in analyzed.keys():
            if func != "global":
                num_accesses += analyzed[func]["accesses"]
                num_cold_starts += analyzed[func]["misses"]

        num_dropped = num_invocations - num_accesses

        cold_start_list.append(num_cold_starts)
        num_accesses_list.append(num_accesses)
        num_dropped_list.append(num_dropped)

    return cold_start_list, num_accesses_list, num_dropped_list



def gen_faascache_hashapps():
    #lambda_data = pd.read_pickle(lambda_data_path)
    #func_mem_dict = lambda_data[0]
    #hashfuncs = func_mem_dict.keys()

    ## Need mapping of hashfunc to hashapp for first day since 
    ## faascache uses first day of data
    #df = pd.read_csv(azure_data)
    #df = df[df.HashFunction.isin(hashfuncs)]
    df = pd.read_pickle("/mnt/femux-results/data/faascache_data/4_cs_results.pickle")
    print(df)

    return df.HashApp.tolist()


def save_faascache_data(cache_sizes, forecasters):

    #faascache_misses, num_invocations, faascache_dropped = gen_faascache_cs(cache_sizes, 2523)
    #faascache_wasted_mem = gen_faascache_wasted_mem(cache_sizes, 2523)

    #df = pd.DataFrame({"Cache Size": cache_sizes,"Misses": faascache_misses, "NumInvocations": num_invocations, "Dropped": faascache_dropped, "Wasted Mem": faascache_wasted_mem})
    #print(df)
    #df.to_pickle(faascache_results.format("faascache"))

    faascache_hashapps = gen_faascache_hashapps()
    for forecaster in forecasters:
        print(faascache_results.format(forecaster))
        forecaster_df = gen_femux_forecasts(15, faascache_hashapps, forecaster)

        forecaster_df[["HashApp", "MemAllocated", "MemoryUsed", "NumColdStarts"]].to_pickle(faascache_results.format(forecaster))


def get_invocation_df():
    invocation_dfs = []
    for i in range(40):
        inv_df = pd.read_pickle(preproc_path.format(i))
        
        inv_df = inv_df[["HashFunction", "InvocationsPerMin", "ExecDurations"]]

        invocation_dfs.append(inv_df)

    return pd.concat(invocation_dfs, ignore_index=True)


def plot_faascache_stats(cache_sizes, num_funcs):
    num_misses, num_accesses, num_dropped = gen_faascache_cs(cache_sizes, num_funcs)
    wasted_mem = gen_faascache_wasted_mem(cache_sizes, num_funcs)

    cache_sizes = [str(cache_size / MB_to_GB) for cache_size in cache_sizes]

    print("Num misses: {}".format(num_misses))
    print("Num dropped: {}".format(num_dropped))
    # Cold Starts
    plt.bar(cache_sizes, num_misses, color="b", label="Missed")
    plt.bar(cache_sizes, num_dropped, bottom=num_misses, color="black", label="Dropped")
    plt.xlabel("Cache Size (GB)")
    plt.ylabel("# Requests")
    plt.legend()
    plt.savefig("cold_starts_{}.pdf".format(num_funcs))
    plt.close()

    print("Wasted GB-s per cache size: {}".format(wasted_mem))
    # Wasted Mem
    plt.bar(cache_sizes, wasted_mem)
    plt.ylabel("Wasted GB-s")
    plt.xlabel("Cache Size (GB)")
    plt.savefig("wasted_mem_{}.pdf".format(num_funcs))
    plt.close()


def gen_femux_stats(num_workers):
    faascache_hashapps = gen_faascache_hashapps()
    df = gen_femux_forecasts(num_workers, faascache_hashapps)

    df["MemWasted"] = df.apply(lambda x : x.MemAllocated - x.MemoryUsed, axis=1)

    num_cold_starts = df.NumColdStarts.sum()
    mem_alloc = df.MemAllocated.sum()
    mem_used = df.MemoryUsed.sum()

    print("FeMux num cold starts: {}".format(num_cold_starts))
    print("FeMux mem wasted: {}".format(mem_alloc - mem_used))


if __name__ == "__main__":
    num_funcs = 2523
    cache_sizes = [200000, 240000, 270000, 300000]
    forecasters = ["Oracle"]#"4_cs", "4_wm", "default"]#, "10_min_keepalive"]
    #plot_faascache_stats(cache_sizes, num_funcs)
    #gen_femux_stats(1)
    save_faascache_data(cache_sizes, forecasters)