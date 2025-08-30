import os
import pandas as pd
import numpy as np
import sys

sys.path.append("../../")
from plotter import plot

from pathlib import Path
from plots.plotters.RUM_plotter import gen_result_df, cold_start_seconds, wasted_GB_seconds

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots" / "azure") + "/"

faascache_dir = data_dir + "faascache/"
hashapp_list_path = data_dir + "transformed_data/small_hashapps.pickle"
cs_wm_forecaster_path = data_dir + "results/_cold_starts_wasted_mem.pickle"
stat_save_path = data_dir + "plotter_data/final_metrics.pickle"
memory_path = data_dir + "preproc_data/memory_data.pickle"
inv_exec_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"

COLD_START_DURATION = 0.808
MS_IN_SEC = 1000
SECONDS_IN_HOUR = 3600

os.makedirs(output_plots_dir + "icebreaker/", exist_ok=True)

# from IceBreaker: cheap  -- t4g, 2vCPU, mem/node=4GB, $/gb-hr=0.0084,  max number of nodes=42 (to match the cost of the costly)
CHEAP_NODE_COST = 0.0084 / SECONDS_IN_HOUR

def plot_final(forecasters, file_desc, data_desc):
    forecasters.extend(["IceBreaker", "10_min_keepalive", "default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics", "MarkovChain_v3", "MarkovChain_v2"])
    results_df = gen_result_df(forecasters, data_desc)
    invocation_df = pd.read_pickle(inv_exec_path.format(12))
    results_df = results_df.merge(invocation_df, on="HashApp", how="left")

    #plot_icebreaker(forecasters, results_df, file_desc)
    #plot_faascache()
    plot_sitw(forecasters, results_df, file_desc)


def plot_icebreaker(forecasters, result_df, file_desc):
    args = dict()
    args["x_label"] = "Normalized Service Time (%)"
    args["y_label"] = "Normalized Keep-alive Cost (%)"# Wasted GB-seconds"
    args["file_name"] = output_plots_dir + "icebreaker/{}_service_time_wasted_mem".format(file_desc)
    args["log"] = False
    args["line"] = False
    close=False

    # pareto frontier
    min_sum = 1000000000000000000000

    total_exec_time = result_df.TotalInvocations.sum() / MS_IN_SEC

    # 10_min keepalive baseline
    result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds("10_min_keepalive", x.NumColdStarts, x.MemoryUsed, 
                                                                        x.MemAllocated, x.SkipBlocks, True), axis=1)

    result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds("10_min_keepalive", x.NumColdStarts, x.MemoryUsed, 
                                                                        x.MemAllocated, x.SkipBlocks, True), axis=1)

    static_service_time = result_df.ColdStartSec.sum() + total_exec_time
    static_ka_cost = result_df.WastedMemTime.sum() * CHEAP_NODE_COST

    forecasters.remove("10_min_keepalive")

    for forecast_num, forecaster in enumerate(forecasters):        
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                            x.MemAllocated, x.SkipBlocks, True), axis=1)

        result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                            x.MemAllocated, x.SkipBlocks, True), axis=1)

        total_service_time = 100 * (result_df.ColdStartSec.sum() + total_exec_time) / static_service_time
        total_ka_cost = 100 * (result_df.WastedMemTime.sum() * CHEAP_NODE_COST) / static_ka_cost

        args["num"] = 0
        if "wm" in forecaster:
            forecaster = "FeMux-Mem"
        elif "4_cs" in forecaster:
            forecaster = "FeMux-CS"
        elif "default" in forecaster:
            forecaster = "FeMux"
        else:
            args["num"] = forecast_num + 1

        forecaster = forecaster.replace("_", " ")
        args["label"] = forecaster
        args["x"] = np.round(total_service_time)
        args["y"] = np.round(total_ka_cost)

        print(forecaster)
        print("Cold start time:" + str(args["x"]))
        print("Wasted Memory Time: " + str(args["y"]))


        if (args["x"] + args["y"] / 100) < min_sum:
            args["x_best"] = args["x"]
            args["y_best"] = args["y"]
            min_sum = args["x"] + args["y"] / 100

        print(forecaster)
        print(print(args["x"] + args["y"]))
        
        plot([], bins=999, args=args, close=close)


def plot_faascache():
    args = dict()
    args["x_label"] = "Number of Cold Starts"
    args["y_label"] = "Wasted GB-seconds"
    args["file_name"] = output_plots_dir + "faascache/cs_num_cs"
    args["line"] = False
    close = False

    cache_sizes = [200000, 240000, 270000, 300000]

    faascache_df = pd.read_pickle(faascache_dir + "faascache_results.pickle")

    for i, cache_size in enumerate(faascache_df["Cache Size"].to_list()):
        faascache_num_cs = (
            faascache_df.iloc[i]["Misses"] + faascache_df.iloc[i]["Dropped"]
        )

        args["label"] = "Faascache {}GB".format(cache_size / 1000)
        args["num"] = 3
        args["x"] = faascache_num_cs
        args["y"] = faascache_df.iloc[i]["Wasted Mem"]
        plot(args) 

    faascache_forecasters = ["4_cs", "4_wm", "default"]

    for forecaster_num, forecaster in enumerate(faascache_forecasters):
        if forecaster_num == len() - 1:
            close = True

        forecaster_df = pd.read_pickle()
        invocation_df = pd.read_pickle(inv_exec_path.format(1))
        forecaster_df = forecaster_df.merge(invocation_df, on="HashApp", how="left")
    
        forecaster_num_cs = forecaster_df.NumColdStarts.sum()
        forecaster_wasted_mem = forecaster_df.MemAllocated.sum() - forecaster_df.MemoryUsed.sum()
        
        args["num"] = 0

        if "wm" in forecaster:
            forecaster = "FeMux-Mem"
        elif "4_cs" in forecaster:
            forecaster = "FeMux-CS"
        elif "default" in forecaster:
            forecaster = "FeMux"
        else:
            args["num"] = forecaster_num + 1
        
        forecaster = forecaster.replace("_", " ")
        args["label"] = forecaster
        args["x"] = forecaster_num_cs
        args["y"] = forecaster_wasted_mem

        plot([], bins=999, args=args, close=close)


def plot_sitw(forecasters, result_df, file_desc):
    args = dict()
    args["x_label"] = "3rd Quartile App Cold Starts (%)"
    args["y_label"] = "Normalized Wasted GB-s (%)"# Wasted GB-seconds"
    args["file_name"] = output_plots_dir + "sitw/{}_75p_app_cs_total_wm".format(file_desc)
    args["log"] = False
    args["line"] = False
    close=False

    os.makedirs(output_plots_dir + "sitw/", exist_ok=True)

    # pareto frontier
    min_sum = 1000000000000000000000

    # 10_min keepalive baseline
    result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds("10_min_keepalive", x.NumColdStarts, x.MemoryUsed, 
                                                                        x.MemAllocated, x.SkipBlocks, True), axis=1)

    result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds("10_min_keepalive", x.NumColdStarts, x.MemoryUsed, 
                                                                        x.MemAllocated, x.SkipBlocks, True), axis=1)

    result_df["ColdStartPct"] = result_df.apply(lambda x : x.ColdStartSec / (x.TotalInvocations * COLD_START_DURATION), axis=1)
    
    static_cs_pct = np.nanpercentile(np.array(result_df.ColdStartPct.tolist()) * 100, 75)
    static_wm = result_df.WastedMemTime.sum()

    print(forecasters)
    # setup sitw data
    for forecast_num, forecaster in enumerate(forecasters):        
        if forecast_num == len(forecasters) - 1:
            close = True

        result_df["ColdStartSec"] = result_df.apply(lambda x : cold_start_seconds(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                            x.MemAllocated, x.SkipBlocks, True), axis=1)

        result_df["WastedMemTime"] = result_df.apply(lambda x : wasted_GB_seconds(forecaster, x.NumColdStarts, x.MemoryUsed, 
                                                                            x.MemAllocated, x.SkipBlocks, True), axis=1)

        result_df["ColdStartPct"] = result_df.apply(lambda x : x.ColdStartSec / (x.TotalInvocations * COLD_START_DURATION), axis=1)


        cs_pct = np.nanpercentile(np.array(result_df.ColdStartPct.tolist()) * 100, 75)
        wasted_mem = 100 * result_df.WastedMemTime.sum() / static_wm
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


if __name__ == "__main__":
    data_desc = "504_100_percent_test"
    file_desc = "FeMux_test"
    #forecasters = ["default_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics", "4_wm_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics", "4_cs_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics"] 
    forecasters = []

    plot_final(forecasters, file_desc, data_desc)