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
inv_exec_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69


def plot_multi(forecasters, data_desc, block_sizes):
    result_df = pd.DataFrame()
    agg_forecasters = []

    for forecaster in forecasters:
        for block_size in block_sizes:
        
            data_path = cs_wm_forecaster_path.replace("/_cold_starts", "/{}/{}_cold_starts".format(data_desc.format(block_size), forecaster))
            forecaster_df = pd.read_pickle(data_path)
            forecaster_df = forecaster_df[['HashApp', 'NumColdStarts', 'NumInvocations', 'MemAllocated', 'MemoryUsed']]

            forecaster_df["SumColdStarts"] = forecaster_df.NumColdStarts.apply(lambda x: sum(x))
            forecaster_df = forecaster_df[forecaster_df.SumColdStarts > 0]
            forecaster_df.drop(["SumColdStarts"], inplace=True, axis=1)

            result_df = update_result_df(result_df, forecaster_df, forecaster + str(block_size))
            agg_forecasters.append(forecaster + str(block_size))
        
    for size in ["small", "medium", "large", "all"]:
        os.makedirs(save_path + "{}/{}/".format(size, "multiblock"), exist_ok=True)
    
    invocation_df = pd.read_pickle(inv_exec_path.format(12))
    result_df = result_df.merge(invocation_df, on="HashApp", how="left")
    
    print(agg_forecasters)
    for size in sizes: 
        plot_agg(agg_forecasters, file_desc, size, result_df)
        plot_sitw(agg_forecasters, result_df, file_desc)


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


def plot_agg(forecasters, file_desc, size, result_df):
    args = dict()
    args["title"] = "{} Applications".format(size)
    args["x_label"] = "Cold Start Seconds"
    args["y_label"] = "Wasted GB-seconds"
    args["file_name"] = save_path + "{}/{}/{}_cs_wm".format(size, "multiblock", file_desc)
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

        result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds(forecaster, x.NumColdStarts), axis=1)

        result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds(forecaster, x.MemoryUsed, 
                                                                            x.MemAllocated), axis=1)

        total_cs_time = sum(result_df.ColdStartSec.to_list())
        total_wm_time = sum(result_df.WastedMemTime.to_list())
        forecaster = forecaster.replace("default_StandardScaler", "")
        forecaster = forecaster.replace("femux", "")
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

        args["num"] = forecast_num

        if (args["x"] + args["y"] / 100) < min_sum:
            args["x_best"] = args["x"]
            args["y_best"] = args["y"]
            min_sum = args["x"] + args["y"] / 100

        print(forecaster)
        print(print(args["x"] + args["y"] / 100))
        

        plot([], bins=999, args=args, close=close)


def cold_start_seconds(forecaster, num_cold_starts):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []
    num_orig_blocks = len(num_cold_starts[forecaster])

    
    if num_orig_blocks == 40:
        num_blocks = 38
    elif num_orig_blocks == 33:
        num_blocks = 31
    elif num_orig_blocks == 23:
        num_blocks = 22
    else:
        num_blocks = 11


    for block_index in range(num_blocks):
        obj_vals.append(num_cold_starts[forecaster][block_index] * COLD_START_DURATION)
    
    return sum(obj_vals)


def wasted_GB_seconds(forecaster, mem_used, mem_allocated):
    """
    num_cold_starts: list[int]
    number of cold starts per block

    mem_used/mem_allocated: list[float]
    memory used per block in GBs

    skip_blocks: list[bool]
    inactive blocks that should be skipped
    """

    obj_vals = []

    num_orig_blocks = len(mem_allocated[forecaster])

    if num_orig_blocks == 40:
        num_blocks = 38
    elif num_orig_blocks == 33:
        num_blocks = 31
    elif num_orig_blocks == 23:
        num_blocks = 22
    else:
        num_blocks = 11

    for block_index in range(num_blocks):#len(mem_allocated[forecaster]) - offset):
        obj_vals.append(mem_allocated[forecaster][block_index] - mem_used[forecaster][block_index])
    
    return sum(obj_vals)


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


def plot_sitw(forecasters, result_df, file_desc):
    os.makedirs(output_plots_dir + "sitw/", exist_ok=True)
    args = dict()
    args["x_label"] = "3rd Quartile App Cold Starts (%)"
    args["y_label"] = "Wasted GB-s (%)"
    args["file_name"] = output_plots_dir + "sitw/{}_75p_app_cs_total_wm".format(file_desc)
    args["log"] = False
    args["line"] = False
    close=False


    # pareto frontier
    min_sum = 1000000000000000000000

    print(forecasters)
    # setup sitw data
    for forecast_num, forecaster in enumerate(forecasters):        
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds(forecaster, x.NumColdStarts), axis=1)

        result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds(forecaster,  x.MemoryUsed, x.MemAllocated), axis=1)

        result_df["ColdStartPct"] = result_df.apply(lambda x : x.ColdStartSec / (x.TotalInvocations * COLD_START_DURATION), axis=1)

        cs_pct = np.nanpercentile(np.array(result_df.ColdStartPct.tolist()) * 100, 75)
        wasted_mem = result_df.WastedMemTime.sum()
        forecaster = forecaster.replace("StandardScaler", "")
        forecaster = forecaster.replace("Density_Linearity_Stationarity_Harmonics", "")
        forecaster = forecaster.replace("_", " ")

        args["label"] = forecaster

        args["x"] = np.round(cs_pct)
        args["y"] = np.round(wasted_mem)

        args["num"] = forecast_num

        if (args["x"] + args["y"] / 100) < min_sum:
            args["x_best"] = args["x"]
            args["y_best"] = args["y"]
            min_sum = args["x"] + args["y"] / 100

        plot([], bins=999, args=args, close=close)


if __name__ == '__main__':
    forecasters = ["default_5_min_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics"]

    file_desc = "multiblock"
    block_sizes = [420, 504, 720, 1440]
    percentage = 100
    data_desc = "{}_100_percent_test"

    sizes = ["all"]

    result_df = plot_multi(forecasters, data_desc, block_sizes)