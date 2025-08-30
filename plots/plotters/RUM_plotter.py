import pickle
import os
import pandas as pd
import numpy as np
import sys
from itertools import chain, combinations

sys.path.append("../../")
from plots.plotters.plotter import plot

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots" / "azure") + "/"

hashapp_list_path = data_dir + "train_test_split/{}_{}_apps.pickle"
cs_wm_forecaster_path = data_dir + "results/_cold_starts_wasted_mem.pickle"
save_path = output_plots_dir + "final_metrics/"
memory_data_path = data_dir + "preproc_data/memory_data.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69


def gen_result_df(forecasters, data_desc):
    result_df = pd.DataFrame()

    for forecaster in forecasters:
        print(forecaster)
        
        data_path = cs_wm_forecaster_path.replace("/_cold_starts", "/{}/{}_cold_starts".format(data_desc, forecaster))
        forecaster_df = pd.read_pickle(data_path)
        forecaster_df = forecaster_df[['HashApp', 'NumColdStarts', 'NumInvocations', 'MemAllocated', 'MemoryUsed']]

        forecaster_df["SumColdStarts"] = forecaster_df.NumColdStarts.apply(lambda x: sum(x))
        forecaster_df = forecaster_df[forecaster_df.SumColdStarts > 0]
        forecaster_df.drop(["SumColdStarts"], inplace=True, axis=1)

        print(forecaster_df)

        result_df = update_result_df(result_df, forecaster_df, forecaster)
        
    result_df["SkipBlocks"] = result_df.MemAllocated.apply(lambda x : gen_skip_blocks(forecasters, x))

    for size in ["small", "medium", "large", "all"]:
        os.makedirs(save_path + "{}/{}/".format(size, data_desc), exist_ok=True)
        os.makedirs(save_path + "winners/{}/{}/".format(size, data_desc), exist_ok=True)

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


def plot_metric(forecasters, data_desc, file_desc, size, metric_calc_func, app_level, result_df):
    metric_name = metric_calc_func.__name__

    print("Plotting {} metric for {} apps".format(metric_name, size))
    
    args = dict()
    args["x_label"] = metric_name.replace("_", " ")
    args["y_label"] = "Fraction of Blocks"
    args["file_name"] = save_path + "{}/{}/{}_{}".format(size, data_desc, file_desc, metric_name)
    args["log"] = True
    args["title"] = "{} Applications".format(size)
    close=False

    result_df = parse_size(result_df, size)

    for forecast_num, forecaster in enumerate(forecasters):        
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["Metric"] = result_df.apply(lambda x : metric_calc_func(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                               x.MemAllocated, x.SkipBlocks, app_level), axis=1)
    
        if app_level:
            vals = list(result_df.Metric.to_list())
            args["y_label"] = "Fraction of Applications"
            args["legend_title"] = ""
        else:
            vals = list(np.concatenate(result_df.Metric.to_list()).flat)
            args["legend_title"] = ""

        forecaster = forecaster.replace("Density_Linearity_Stationarity_Harmonics", "")
        forecaster = forecaster.replace("_", " ")
        forecaster = ''.join([i for i in forecaster if not i.isdigit()])
        #forecaster = "IceBreaker-AC" if "FFT" in forecaster else forecaster
        forecaster = forecaster.replace("StandardScaler", "")
        args["label"] = forecaster
        args["forecaster_num"] = forecast_num
        plot(vals, bins=999, args=args, close=close)


def plot_agg(forecasters, data_desc, file_desc, size, weight_mode, result_df):
    args = dict()
    args["title"] = "{} Applications".format(size)
    args["x_label"] = "Cold Start Seconds"
    args["y_label"] = "Wasted GB-seconds"
    args["file_name"] = save_path + "{}/{}/{}_cs_wm".format(size, data_desc, file_desc)
    args["log"] = False
    args["line"] = True
    close=False

    # pareto frontier
    min_sum = 1000000000000000000000
    
    print("Plotting aggregate for {} apps".format(size))
        
    result_df = parse_size(result_df, size)
    
    for forecast_num, forecaster in enumerate(forecasters):        
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                            x.MemAllocated, x.SkipBlocks, True), axis=1)

        result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                            x.MemAllocated, x.SkipBlocks, True), axis=1)

        total_cs_time = sum(result_df.ColdStartSec.to_list())
        total_wm_time = sum(result_df.WastedMemTime.to_list())
        forecaster = forecaster.replace("default_StandardScaler", "")
        #forecaster = forecaster.replace("femux", "")
        forecaster = forecaster.replace("Density", "D")
        forecaster = forecaster.replace("Linearity", "L")
        forecaster = forecaster.replace("Stationarity", "S")
        forecaster = forecaster.replace("Harmonics", "H")
        forecaster = forecaster.replace("_", " ")
        forecaster = forecaster.replace("no_ema_", " ")

        args["label"] = forecaster

        args["x"] = np.round(total_cs_time)
        args["y"] = np.round(total_wm_time)

        print(forecaster)
        print("Cold start time:" + str(args["x"]))
        print("Wasted Memory Time: " + str(args["y"]))

        if weight_mode: 
            print("adding weights")
            args["x"] *= COLD_START_DURATION
            args["y"] *= WASTED_MEMORY_WEIGHT
            #if "420" not in forecaster:
            #    args["x"] *= 1.015625
            #    args["y"] *= 1.015625

            args["x_label"] = "Weighted Cold Start Seconds"
            args["y_label"] = "Weighted Wasted GB-seconds"


        args["num"] = forecast_num

        if (args["x"] + args["y"] / 100) < min_sum:
            args["x_best"] = args["x"]
            args["y_best"] = args["y"]
            min_sum = args["x"] + args["y"] / 100

        print(forecaster)
        print(print(args["x"] + args["y"] / 100))
        

        plot([], bins=999, args=args, close=close)


def plot_obj_func_winners(forecasters, size, data_desc, file_desc, result_df):
    args = dict()
    args["title"] = "{} Applications".format(size)
    args["x_label"] = "Forecaster"
    args["y_label"] = "Objective Value"
    args["file_name"] = save_path + "/winners/{}/{}/{}_obj_winners".format(size, data_desc, file_desc)
    args["boxplot"] = True
    args["forecaster_labels"] = []

    result_df = parse_size(result_df, size)

    result_df["ObjectiveWinners"] = result_df.apply(lambda x : gen_winner_list(x.NumColdStarts, x.MemoryUsed, 
                                                                               x.MemAllocated, forecasters), axis=1)

    forecaster_wins = list(np.concatenate(result_df.ObjectiveWinners.to_list()))
    forecaster_vals = []

    for forecaster in forecasters:
        cur_wins = [float(forecaster_win[1]) for forecaster_win in forecaster_wins if forecaster_win[0] == forecaster]
        forecaster_vals.append(cur_wins)
        args["forecaster_labels"].append("{} ({})".format(forecaster, len(cur_wins))) 

    plot(forecaster_vals, 99, args, close=True)


def gen_winner_list(num_cold_starts, mem_used, mem_alloc, forecasters):
    num_blocks = len(num_cold_starts[forecasters[0]])
    winner_list = []


    for block_index in range(num_blocks - 1):
        best_result = calc_objective_function(forecasters[0], block_index, 
                                              num_cold_starts, mem_used, mem_alloc)
        best_forecaster = forecasters[0]
        results = [best_result]

        for forecaster in forecasters[1:]:
            cur_result = calc_objective_function(forecaster, block_index, 
                                                 num_cold_starts, mem_used, mem_alloc)

            if cur_result < best_result:
                best_result = cur_result
                best_forecaster = forecaster

            results.append(cur_result)

        # flag this block as a skip if all of the forecasters are 0 (indicating nothing happened) 
        improvement_over_median = np.percentile(results, 50) - best_result
        winning_forecaster = best_forecaster if improvement_over_median > MIN_METRIC else "Skip"
        winner_list.append([winning_forecaster, improvement_over_median])

    return winner_list


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
    for block_index in range(33):#len(mem_allocated[forecaster]) - 1):
        if skip_blocks[block_index]:
            continue

        obj_vals.append(calc_objective_function(forecaster, block_index, num_cold_starts, 
                                                mem_used, mem_allocated))

    return sum(obj_vals) if app_level else obj_vals


def cold_start_seconds(forecaster, num_cold_starts, mem_used, mem_allocated, skip_blocks, app_level):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []

    for block_index in range(33):#len(mem_allocated[forecaster]) - offset):
        if skip_blocks[block_index]:
            continue
        
        obj_vals.append(num_cold_starts[forecaster][block_index] * COLD_START_DURATION)
    
    return sum(obj_vals) if app_level else obj_vals


def wasted_GB_seconds(forecaster, num_cold_starts, mem_used, mem_allocated, skip_blocks, app_level):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []

    for block_index in range(33):#len(mem_allocated[forecaster]) - offset):
        if skip_blocks[block_index]:
            continue
        
        obj_vals.append(mem_allocated[forecaster][block_index] - mem_used[forecaster][block_index])
    
    return sum(obj_vals) if app_level else obj_vals


def parse_size(df, size):
    if size == "all":
        return df

    overall_list = [] 
    for data in ["training", "test"]:
        list_path = hashapp_list_path.format(size, data)
    
        with open(list_path, "rb") as f:
            size_list = pickle.load(f)
            overall_list.extend(size_list)

    return df[df.HashApp.isin(overall_list)] 

def plot_final(forecasters, file_desc, data_desc, sizes, weight_mode, femux_features=None, app_level=False):
    result_df = gen_result_df(forecasters, data_desc)

    for size in sizes:
        #for func in [cold_start_seconds, wasted_GB_seconds, objective_function]:
        #    plot_metric(forecasters, data_desc, file_desc, size, func, app_level, result_df)
        
        plot_agg(forecasters, data_desc, file_desc, size, weight_mode, result_df)
        #plot_obj_func_winners(forecasters, size, data_desc, file_desc, result_df)


def add_mem_values(df):
    mem_df = pd.read_pickle(memory_data_path)

    df = df.merge(mem_df, on="HashApp", how="left")

    return df


def powerset(features):
        return list(chain.from_iterable(combinations(features, r) for r in range(len(features)+1)))[1:]

if __name__ == '__main__':
    forecasters = []

    base_name = "default_no_ema_StandardScaler_femux_{}"

    features = ["Density", "Linearity", "Stationarity", "Harmonics"]
    
    for features in powerset(features):
        if len(features) > 2:
            continue 
        features = list(features)
        feature_names = "_".join(features)
        forecasters.append(base_name.format(feature_names))

    forecasters = ["default_new_kmeans_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics", "4_cs_new_kmeans_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics", "default_new_kmeans_PowerTransformer_femux_Density_Linearity_Stationarity_Harmonics"]#, "MarkovChain_v3", "FFT_10", "Holt", "ExpSmoothing", "AR", "SETAR", "5_min_keepalive", "10_min_keepalive"]

    file_desc = "femux_comp"
    block_size = 504
    percentage = 100
    data_desc = "{}_{}_percent_test".format(block_size, percentage)
    weight_mode = False
    app_level = False

    sizes = ["all"]

    plot_final(forecasters, file_desc, data_desc, sizes, weight_mode, app_level=app_level) 