import gc
import pickle
import pandas as pd
import numpy as np
import sys
from os import listdir
sys.path.append("../../code/")
from concurrent.futures import ProcessPoolExecutor
import results.utils as utils
from time import strftime
from math import ceil
from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
conc_forecast_path = data_dir + "forecaster_data/concurrency/conc_forecast_{}_{:02d}.pickle"
exec_path = data_dir + "preproc_data/app_exec_time_data.pickle"
result_path = data_dir + "results/504_100_percent_test/Aquatope_cold_starts_wasted_mem.pickle"

NUM_TRAINING_DAYS = 7
NUM_AQUATOPE_INPUT_STEPS = 48
BLOCK_SIZE = 504
STATIC_KEEPALIVE_WINDOW = 10
MINUTES_PER_DAY = 1440
START_INDEX = (BLOCK_SIZE * 21) - NUM_TRAINING_DAYS * MINUTES_PER_DAY - NUM_AQUATOPE_INPUT_STEPS
END_INDEX = (BLOCK_SIZE * 34) - NUM_TRAINING_DAYS * MINUTES_PER_DAY - NUM_AQUATOPE_INPUT_STEPS

SECONDS_PER_MIN = 60
MB_to_GB = 1000
MIN_CONC = 1.6e-05
MIN_FORECAST = MIN_CONC / 2 

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69
SECONDS_IN_DAY = 86400

def gen_forecast_results(num_workers, forecaster, splits):
    forecast_window = [120] * num_workers

    df = get_aquatope_forecasts(splits)
    print(df)
    df = preproc_data(df)
    print(df)
        
    if forecaster == "Oracle":
        pass
    elif "femux" in forecaster:
        del df["ForecastedValues"]
        df = utils.set_forecaster(df, "femux", "processed_testing_df_{}_markov_v3_StandardScaler_504_100_percent_test_Density_Linearity_Stationarity_Harmonics.pickle".format(forecaster)) 

    dfs = np.array_split(df, num_workers)
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(gen_multiproc_results, dfs, [forecaster] * num_workers, forecast_window)

    result_df = pd.concat(results, ignore_index=True)

    result_df.to_pickle(result_path)

def gen_multiproc_results(df, forecaster, forecast_window):
    if forecaster == "Oracle":
        forecast_window = 0
        df["ForecastedValues"] = df.TransformedValues.apply(lambda x : [[x[i]] for i in range(forecast_window, len(x))])


    print("Starting cold starts", strftime("%H:%M:%S"))
    df[["NumColdStarts", "NumInvocations"]] = df.apply(lambda x : gen_cold_starts(x.ForecastedValues,
                                                                    x.TransformedValues,
                                                                    x.ContainerInvocationsPerMin), axis=1)

    gc.collect()
    print("Starting memory calculations", strftime("%H:%M:%S"))

    df[["MemAllocated", "MemoryUsed"]] = df.apply(lambda x : gen_mem_usage(x.ForecastedValues,
                                                                x.TransformedValues,
                                                                x.AverageMemUsage[0],
                                                                x.ContainerInvocationsPerMin), axis=1)

    return df


def get_aquatope_forecasts(splits):
    dfs = []
    for split in splits:
        df = pd.read_pickle(conc_forecast_path.format("aquatope", split))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={"TransformedValues": "ForecastedValues"}, inplace=True)
    df["ForecastedValues"] = df.ForecastedValues.apply(lambda x : [[y] for y in x])
    return df[["HashApp", "ForecastedValues"]]


def preproc_data(df):
    """ Merge memory, traffic, and container data from days 8-12 into the dataframe.
    """
    mem_inv_df = utils.gather_data_for_results("test", 100)
    mem_inv_df = utils.clean_data(mem_inv_df, 504)
    
    # remove the first 7 days of data since we trained on 7 days
    mem_inv_df["TransformedValues"] = mem_inv_df.TransformedValues.apply(lambda x : x[(NUM_TRAINING_DAYS * MINUTES_PER_DAY) + NUM_AQUATOPE_INPUT_STEPS:])
    mem_inv_df["ContainerInvocationsPerMin"] = mem_inv_df.ContainerInvocationsPerMin.apply(lambda x : x[(NUM_TRAINING_DAYS * MINUTES_PER_DAY) + NUM_AQUATOPE_INPUT_STEPS:])
    mem_inv_df["AverageMemUsage"] = mem_inv_df.AverageMemUsage.apply(lambda x : x[NUM_TRAINING_DAYS:])
    df = pd.merge(df, mem_inv_df, on="HashApp")
    df = df[df.TransformedValues.apply(lambda x : sum(x) > 0)]
    
    return df



def gen_cold_starts(forecasted_vals, trace, invocations):
    """ Calculates the number of cold starts for each block

    forecasted_vals: list[float]

    trace: list[float]

    num_invocations: int
    """
    cold_start_sum = 0
    container_invocation_sum = 0

    for cur_minute in range(START_INDEX, END_INDEX):
        num_predicted_containers = 0 if forecasted_vals[cur_minute][0] < MIN_FORECAST else ceil(forecasted_vals[cur_minute][0])
        num_containers = 0 if trace[cur_minute] < MIN_CONC else ceil(trace[cur_minute])
        container_invocation_sum += invocations[cur_minute]

        num_cold_starts = max(0, num_containers - num_predicted_containers)

        # Average concurrency can be positive due to invocations from previous minutes,
        # so we can only count cold starts for invocations that occur in the current minute
        cold_start_sum += min(num_cold_starts, invocations[cur_minute])

    return pd.Series([cold_start_sum, container_invocation_sum])


def gen_mem_usage(forecasted_vals, trace, mem_val, invocations):
    """ Calculates the resource utilization for each block

    forecasted_vals: list[float]

    trace: list[float]

    mem_vals: list[int]
    Memory usage per instance of application in MB
    """
    mem_usage = mem_val / MB_to_GB
    
    total_mem_allocated = 0
    total_mem_used = 0

    for cur_minute in range(START_INDEX, END_INDEX):
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


if __name__ == "__main__":
    num_workers = 12


    splits = list(range(9)) + [10,13, 16, 20, 21] + list(range(25, 49)) + list(range(50,75)) + list(range(75, 99))
    gen_forecast_results(num_workers, "aquatope", splits)