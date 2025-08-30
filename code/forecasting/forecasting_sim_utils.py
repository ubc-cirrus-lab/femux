import gc
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append("..")
from results.gen_results import calc_mem_alloc

data_dir = str(Path(__file__).resolve().parents[2] / "data") + "/azure/"

SECONDS_PER_MIN = 60
MIN_CONC = 1.6e-05
MIN_FORECAST = MIN_CONC / 2

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69


def gen_obj_val(weight_mode, forecasted_val, trace_val, invocation_val, mem_usage, exec_time):
    """Objective function takes a weighted sum of cold start duration and wasted memory
    """
    wasted_mem = calc_wasted_mem(forecasted_val, trace_val, invocation_val, mem_usage)

    num_cold_starts = calc_cold_starts(forecasted_val, trace_val, invocation_val)

    cold_start_time_weight = 4 * COLD_START_TIME_WEIGHT if weight_mode == "4_cs" else COLD_START_TIME_WEIGHT
    wasted_mem_weight = 4 * WASTED_MEMORY_WEIGHT if weight_mode == "4_wm" else WASTED_MEMORY_WEIGHT

    if weight_mode == "exec":
        obj_val = COLD_START_TIME_WEIGHT * COLD_START_DURATION * num_cold_starts / exec_time + WASTED_MEMORY_WEIGHT * wasted_mem
    else:
        obj_val = cold_start_time_weight * COLD_START_DURATION + wasted_mem_weight * wasted_mem
    
    obj_val = 0 if obj_val < MIN_METRIC else obj_val

    return obj_val


def calc_cold_starts(forecasted_val, trace_val, invocation_val):
    """ Calculates the number of cold starts for each block

    forecasted_vals: list[float]

    trace: list[float]

    num_invocations: int

    """
    num_predicted_containers = 0 if forecasted_val < MIN_FORECAST else np.ceil(forecasted_val)
    num_containers = 0 if trace_val < MIN_CONC else np.ceil(trace_val)

    num_missing_containers = max(0, num_containers - num_predicted_containers)
    
    # Average concurrency can be positive due to invocations from previous minutes,
    # so we can only count cold starts for invocations that occur in the current minute
    num_cold_starts = min(num_missing_containers, invocation_val)

    return num_cold_starts


def calc_wasted_mem(forecasted_val, trace_val, invocation_val, mem_usage):
    """ Calculates the resource utilization for each block

    forecasted_vals: float

    trace: float

    invocation_val: float

    mem_val: float
    Memory usage per instance of application in GBs
    """
    trace_val = 0 if trace_val < MIN_CONC else trace_val
    
    num_predicted_containers = 0 if forecasted_val < MIN_FORECAST else np.ceil(forecasted_val)
    num_actual_containers = np.ceil(trace_val)

    # predicted containers are kept alive throughout the whole minute
    predicted_mem_alloc = num_predicted_containers * mem_usage * SECONDS_PER_MIN

    # when FeMux underpredicts the number of containers, include the memory used during execution for
    # containers that experience cold starts (first n invocations within the minute where n 
    # is the number of cold starts). These are kept alive until the end of the minute.
    remaining_mem_alloc = calc_mem_alloc(num_actual_containers, num_predicted_containers, 
                                    mem_usage, invocation_val)
    
    # memory utilization only counts the memory used during execution
    mem_used = trace_val * mem_usage * SECONDS_PER_MIN

    mem_allocated = max(remaining_mem_alloc + predicted_mem_alloc, mem_used)

    wasted_memory = mem_allocated - mem_used

    return wasted_memory