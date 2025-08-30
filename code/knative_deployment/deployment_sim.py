import gc
import pandas as pd
import numpy as np
import sys
import os

sys.path.append("..")
import results.utils as utils
from concurrent.futures import ProcessPoolExecutor
from time import strftime
from math import ceil, isclose
from pathlib import Path

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

deployment_data_dir = data_dir + "knative_deployment_data/"
cold_start_save_path = deployment_data_dir + "{}_sim_result.pickle"

NUM_DAYS = 1
MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
MB_to_GB = 1000
MIN_CONC = 1.6e-5
MIN_FORECAST = MIN_CONC / 2 

MIN_METRIC = 1e-6

RESULT_COLS = ["HashApp", "NumColdStarts",  "MemAllocated", "MemoryUsed"]

def gen_results(forecaster, forecast_window, forecast_len, num_workers, femux_path):
    """
    file_subscript: str
    
    forecaster: str

    forecast_window: int
    num past elements

    forecast_len: int
    how many future timesteps are forecasted at each timestep

    num_workers: int
    
    data: str
    "train" for training data, "test" for test data, "deployment" for knative deployment

    data_percentage: int
    percentage of data to sample from train or test

    femux_features: str
    underscore-separated list of femux feature (e.g., "EMADensity_EMALinearity")
    """
    save_path = cold_start_save_path.format(forecaster)
    df = utils.gather_data_for_results("deployment")
    
    print("adding forecaster values for {}".format(forecaster), strftime("%H:%M:%S"))
    df = utils.set_forecaster(df, forecaster, femux_path)
    
    print("generating results".format(forecaster), strftime("%H:%M:%S"))
    dfs = np.array_split(df, num_workers)
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(gen_multiproc_results, dfs, [forecaster] * num_workers, 
                                [forecast_window] * num_workers, [forecast_len] * num_workers)

    result = pd.concat(results, ignore_index=True)

    print("Saving results to {}".format(save_path))
    result[RESULT_COLS].to_pickle(save_path)

    del result
    del results
    gc.collect()


def gen_multiproc_results(df, forecaster, forecast_window, forecast_len):

    print("Starting cold starts", strftime("%H:%M:%S"))
    df["NumColdStarts"] = df.apply(lambda x : gen_cold_starts(forecaster, x.ForecastedValues,
                                                                    x.TransformedValues,
                                                                    x.ContainerInvocationsPerMin,
                                                                    forecast_window,
                                                                    forecast_len), axis=1)

    gc.collect()
    print("Starting memory calculations", strftime("%H:%M:%S"))

    df[["MemAllocated", "MemoryUsed"]] = df.apply(lambda x : gen_mem_usage(forecaster,
                                                                x.ForecastedValues,
                                                                x.TransformedValues,
                                                                x.AverageMemUsage,
                                                                x.ContainerInvocationsPerMin,
                                                                forecast_window,
                                                                forecast_len), axis=1)
 
    return df


def gen_cold_starts(forecaster, forecasted_vals, trace, invocations, 
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
    if forecast_window > 0:
        static_ka = 1 if forecaster == "Default_Knative" else 10
        static_keepalive = utils.gen_static_keepalive(trace[:forecast_window], static_ka)
        forecasted_vals = np.concatenate((static_keepalive, forecasted_vals), dtype=object)

    # for femux we only forecast until the 34th block since it forecasts up to block
    # crossings.
    end_minute = MINUTES_PER_DAY

    cold_starts_per_min = np.empty(end_minute)

    for cur_minute in range(end_minute):
        num_predicted_containers = 0 if forecasted_vals[cur_minute][0] < MIN_FORECAST else ceil(forecasted_vals[cur_minute][0])
        num_containers = 0 if trace[cur_minute] < MIN_CONC else ceil(trace[cur_minute])

        num_cold_starts = max(0, num_containers - num_predicted_containers)

        # Average concurrency can be positive due to invocations from previous minutes,
        # so we can only count cold starts for invocations that occur in the current minute
        cold_starts = min(num_cold_starts, invocations[cur_minute])

        cold_starts_per_min[cur_minute] = cold_starts

    return pd.Series([cold_starts_per_min])


def gen_mem_usage(forecaster, forecasted_vals, trace, mem_vals, invocations,
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

    
    mem_vals = [mem_val / MB_to_GB for mem_val in mem_vals]

    # use static keepalive for timesteps before forecasting starts
    if forecast_window > 0:
        static_ka = 1 if forecaster == "Default_Knative" else 10
        static_keepalive = utils.gen_static_keepalive(trace[:forecast_window], static_ka)
        forecasted_vals = np.concatenate((static_keepalive, forecasted_vals), dtype=object)

    # memory stops at day 12, but traces go until day 14, so we have to stop early
    end_minute = MINUTES_PER_DAY 
    cur_day = 0

    mem_allocated_per_min = np.empty(end_minute)
    mem_used_per_min = np.empty(end_minute)
    block_index = 0

    for cur_minute in range(end_minute):
        if cur_minute % MINUTES_PER_DAY == 0:
            mem_usage = mem_vals[cur_day]
            cur_day += 1

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
        mem_allocated = max(mem_allocated, mem_used)
        mem_used = mem_used 

        mem_allocated_per_min[cur_minute] = mem_allocated
        mem_used_per_min[cur_minute] = mem_used


    return pd.Series([mem_allocated_per_min, mem_used_per_min])


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


if __name__ == '__main__':
    
    forecasters = ["Default_Knative", "femux"]
    
    block_size = 504
    num_workers = 12
    forecast_len = 1
    femux_features = "Density_Linearity_Stationarity_Harmonics"
    femux_transformer = "processed_testing_df_kmeans_default_StandardScaler_504_100_percent_Density_Linearity_Stationarity_Harmonics"

    femux_path = "processed_testing_df_kmeans_default_StandardScaler_504_100_percent_Density_Linearity_Stationarity_Harmonics.pickle"

    for forecaster in forecasters:
        forecast_window = 120

        if forecaster == "IceBreaker" or forecaster == "FFT_10":
            forecast_window = 60

        gen_results(forecaster, forecast_window, forecast_len, num_workers, femux_path)
        gc.collect()