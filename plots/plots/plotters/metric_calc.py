import numpy as np
MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69


def calc_objective_function(forecaster, block_index, num_cold_starts, mem_used, mem_alloc):

    cold_start_time = num_cold_starts[forecaster][block_index] * COLD_START_DURATION
    wasted_memory = mem_alloc[forecaster][block_index] - mem_used[forecaster][block_index]

    obj_func = COLD_START_TIME_WEIGHT * cold_start_time + WASTED_MEMORY_WEIGHT * wasted_memory

    return obj_func


def objective_function(forecaster, num_cold_starts, mem_used, mem_allocated, skip_blocks, app_level):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []
    for block_index in range(33):
        if skip_blocks[block_index]:
            continue

        obj_vals.append(calc_objective_function(forecaster, block_index, num_cold_starts, 
                                                mem_used, mem_allocated))

    return sum(obj_vals) if app_level else obj_vals


def cold_start_seconds(forecaster, num_cold_starts, mem_used, mem_allocated, skip_blocks, app_level, start=0, end=33):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []

    for block_index in range(start, end):
        if skip_blocks[block_index]:
            continue
        
        obj_vals.append(num_cold_starts[forecaster][block_index] * COLD_START_DURATION)
    
    return sum(obj_vals) if app_level else np.array(obj_vals)


def wasted_GB_seconds(forecaster, num_cold_starts, mem_used, mem_allocated, skip_blocks, app_level, start=0, end=33):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []

    for block_index in range(start, end):
        if skip_blocks[block_index]:
            continue
        
        obj_vals.append(mem_allocated[forecaster][block_index] - mem_used[forecaster][block_index])
    
    return sum(obj_vals) if app_level else np.array(obj_vals)
