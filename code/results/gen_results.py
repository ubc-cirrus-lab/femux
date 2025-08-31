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

cold_start_save_path = data_dir + "results/_cold_starts_wasted_mem.pickle"

NUM_DAYS = 12
MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
MB_to_GB = 1000
MIN_CONC = 1.6e-5
MIN_FORECAST = MIN_CONC / 2 

MIN_METRIC = 1e-6

RESULT_COLS = ["HashApp", "NumColdStarts", "NumInvocations", "MemAllocated", "MemoryUsed"]

def gen_results(file_subscript, forecaster, forecast_window, forecast_len, num_workers, data, 
                data_percentage, block_size, femux_path):
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
    """
    print("starting {} for {}".format(forecaster, file_subscript))
    
    df = utils.gather_data_for_results(data, data_percentage)
    df = utils.clean_data(df, block_size)
    save_path = cold_start_save_path.replace("/_cold", "/{}/{}_cold".format(file_subscript, forecaster))
    os.makedirs(data_dir + "results/{}".format(file_subscript), exist_ok = True)
    
    print("adding forecaster values for {}".format(forecaster), strftime("%H:%M:%S"))
    df = utils.set_forecaster(df, forecaster, femux_path)
    
    if df.isnull().values.any():
        prev_len = len(df)
        df = df.dropna()
        num_missing = prev_len - len(df)

        raise Exception("Missing {} results for selected dataset".format(num_missing))

    print("generating results".format(forecaster), strftime("%H:%M:%S"))
    dfs = np.array_split(df, num_workers)
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(gen_multiproc_results, dfs, [forecaster] * num_workers, 
                                [forecast_window] * num_workers, [forecast_len] * num_workers, 
                                [block_size] * num_workers)

    result = pd.concat(results, ignore_index=True)

    print("Saving results to {}".format(save_path))
    result[RESULT_COLS].to_pickle(save_path)

    del result
    del results
    gc.collect()


def gen_multiproc_results(df, forecaster, forecast_window, forecast_len, block_size):
    if forecaster == "Oracle":
        forecast_window = 0
        df["ForecastedValues"] = df.TransformedValues.apply(lambda x : [[x[i]] for i in range(forecast_window, len(x))])

    print("Starting cold starts", strftime("%H:%M:%S"))
    df[["NumColdStarts", "NumInvocations"]] = df.apply(lambda x : gen_cold_starts(forecaster, x.ForecastedValues,
                                                                    x.TransformedValues,
                                                                    x.ContainerInvocationsPerMin,
                                                                    forecast_window,
                                                                    forecast_len,
                                                                    block_size), axis=1)

    gc.collect()
    print("Starting memory calculations", strftime("%H:%M:%S"))

    df[["MemAllocated", "MemoryUsed"]] = df.apply(lambda x : gen_mem_usage(forecaster,
                                                                x.ForecastedValues,
                                                                x.TransformedValues,
                                                                x.AverageMemUsage,
                                                                x.ContainerInvocationsPerMin,
                                                                forecast_window,
                                                                forecast_len,
                                                                block_size), axis=1)
 
    return df


def gen_cold_starts(forecaster, forecasted_vals, trace, invocations, 
                        forecast_window, forecast_len, block_size):
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
        static_keepalive = utils.gen_static_keepalive(trace[:forecast_window])
        forecasted_vals = np.concatenate((static_keepalive, forecasted_vals), dtype=object)

    cold_start_sum = 0
    container_invocation_sum = 0
        
    # for femux we only forecast until the 34th block since it forecasts up to block
    # crossings.
    num_femux_elements = (NUM_DAYS * MINUTES_PER_DAY // block_size) * block_size
    end_minute = num_femux_elements - forecast_len if "femux" in forecaster else NUM_DAYS * MINUTES_PER_DAY - forecast_len

    cold_starts_per_block = np.empty(end_minute // block_size)
    invocations_per_block = np.empty(end_minute // block_size)
    block_index = 0

    for cur_minute in range(end_minute):
        num_predicted_containers = 0 if forecasted_vals[cur_minute][0] < MIN_FORECAST else ceil(forecasted_vals[cur_minute][0])
        num_containers = 0 if trace[cur_minute] < MIN_CONC else ceil(trace[cur_minute])
        container_invocation_sum += invocations[cur_minute]
 
        num_cold_starts = max(0, num_containers - num_predicted_containers)

        # Average concurrency can be positive due to invocations from previous minutes,
        # so we can only count cold starts for invocations that occur in the current minute
        cold_start_sum += min(num_cold_starts, invocations[cur_minute])

        if (cur_minute + 1) % block_size == 0:
            cold_starts_per_block[block_index] = cold_start_sum
            invocations_per_block[block_index] = container_invocation_sum           
            cold_start_sum = 0
            container_invocation_sum = 0
            block_index += 1

    return pd.Series([cold_starts_per_block, invocations_per_block])


def gen_mem_usage(forecaster, forecasted_vals, trace, mem_vals, invocations,
                    forecast_window, forecast_len, block_size):
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

    if forecast_window > 0:
        static_keepalive = utils.gen_static_keepalive(trace[:forecast_window])
    
        # use static keepalive for timesteps before forecasting starts
        forecasted_vals = np.concatenate((static_keepalive, forecasted_vals), dtype=object)
    
    # memory stops at day 12, but traces go until day 14, so we have to stop early
    num_femux_elements = (NUM_DAYS * MINUTES_PER_DAY // block_size) * block_size
    end_minute = num_femux_elements - forecast_len if "femux" in forecaster else NUM_DAYS * MINUTES_PER_DAY - forecast_len

    cur_day = 0

    mem_allocated_per_block = np.empty(end_minute // block_size)
    mem_used_per_block = np.empty(end_minute // block_size)
    total_mem_allocated = 0
    total_mem_used = 0
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
        total_mem_allocated += max(mem_allocated, mem_used)
        total_mem_used += mem_used 

        if (cur_minute + 1) % block_size == 0:
            mem_allocated_per_block[block_index] = total_mem_allocated
            mem_used_per_block[block_index] = total_mem_used
            total_mem_allocated = 0
            total_mem_used = 0
            block_index += 1

    return pd.Series([mem_allocated_per_block, mem_used_per_block])


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
    forecast_len = 1
    
    forecasters = ["AR", "FFT_10", "MarkovChain", "10_min_keepalive", "5_min_keepalive", "IceBreaker"] #("ExpSmoothing", "Holt", "SETAR")
    
    data_mode = "test"
    data_percentage = 100
    block_size = 504
    num_workers = 48
    classification_model = "kmeans"
    femux_features = "Density_Linearity_Stationarity_Harmonics"
    femux_transformer = "None"
    femux_weight_mode = "default"

    femux_path =  "processed_testing_df_{}_{}_{}_{}_{}_percent_{}.pickle".format(classification_model, femux_weight_mode, femux_transformer, 
                                                                                block_size, data_percentage, femux_features)
        
    for forecaster in forecasters:
        for block_size in [504]:
            for data_mode in ["train", "test"]:
                file_subscript = "{}_{}_percent_{}".format(block_size, data_percentage, data_mode)
                forecast_window = 120

                if forecaster == "IceBreaker" or forecaster == "FFT_10":
                    forecast_window = 60
                elif "IdleTime" in forecaster or "HybridHist" in forecaster:
                    forecast_window = 0

                gen_results(file_subscript, forecaster, forecast_window, forecast_len, num_workers, data_mode, data_percentage, block_size, femux_path)
                gc.collect()
