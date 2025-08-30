import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import deque 

sys.path.append("../../code/")
from results.utils import add_transform_values


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 14

data_dir = str(Path(__file__).parents[2] / "data"/ "azure") + "/"
output_plots_dir = str(Path(__file__).parents[1] / "output_plots" / "carbon") + "/"


cs_wm_forecaster_path = data_dir + "results/420_100_percent_train/{}_cold_starts_wasted_mem.pickle"
exec_data_file = data_dir + "preproc_data/app_total_inv_exec_12_days.pickle"
memory_data_file = data_dir + "preproc_data/memory_data.pickle"
max_memory_data_file = data_dir + "preproc_data/max_memory_data.pickle"

SECONDS_IN_HOUR = 3600
mb_per_core = 1769
cs_duration = 1.07 # from how does it function socc '23
mem_power_draw = 0.3725 # W/GB
idle_cpu_power_draw = 0.75 # W / core
utilized_cpu_power_draw = 3.5 # W / core

os.makedirs(output_plots_dir, exist_ok=True)



def plot_carbon_ratio_vs_cs_duration(forecasters):
    axs = plt.figure(figsize=(6.8, 2.8), dpi=300, constrained_layout=True).subplots(1,2)
    exec_mem_cores_df = preproc_df()

    cs_pct_per_forecaster = []

    forecaster_names = ["1-min", "5-min", "10-min"]

    for i, forecaster in enumerate(forecasters):
        cs_wm_df = pd.read_pickle(cs_wm_forecaster_path.format(forecaster))
        df = exec_mem_cores_df.merge(cs_wm_df, on="HashApp", how="left")
        
        df[["ExeckWh", "IdlekWh"]] = df.apply(lambda x : calc_carbon(x.TotalExec, x.IdleSeconds, x.MemoryUsed, x.MemAllocated, x.NumCores), axis=1)
        df["CarbonRatio"] = df.IdlekWh / df.ExeckWh
        df["ColdStartPct"] = df.apply(lambda x : 100 * sum(x.NumColdStarts) / sum(x.InvocationsPerMin), axis=1)
        
        plot_cdf(axs[0], df.CarbonRatio.tolist(), label="Carbon Ratio (Idle / Execution)", forecaster=forecaster_names[i], filename="cs_cdf")
        plot_cdf(axs[1], df.ColdStartPct.tolist(), label="Cold Start (%)", forecaster = forecaster_names[i], filename="cs_cdf", legend=False, log=False)
 
    plt.tight_layout()
    plt.close()


def plot_max_mem_vs_avg():
    exec_mem_cores_df = preproc_df()

    ax = plt.figure(figsize=(5.8, 2.8), dpi=300, constrained_layout=True).subplots(1,1)

    # Dynamic Alloc
    cs_wm_df = pd.read_pickle(cs_wm_forecaster_path.format("5_min_keepalive"))
    df = exec_mem_cores_df.merge(cs_wm_df, on="HashApp", how="left")        
    df[["ExeckWh", "IdlekWh"]] = df.apply(lambda x : calc_carbon(x.TotalExec, x.IdleSeconds, x.MemoryUsed, x.MemAllocated, x.NumCores), axis=1)
    df["TotalkWh"] = df.apply(lambda x : x.IdlekWh + x.ExeckWh, axis=1)
    dynamic_alloc_kwh = df.TotalkWh.tolist()
    
    exec_mem_cores_df = preproc_df(max_mem=True)
    cs_wm_df = pd.read_pickle(cs_wm_forecaster_path.format("10_min_keepalive").replace("mem", "max_mem"))
    df = exec_mem_cores_df.merge(cs_wm_df, on="HashApp", how="left")        
    df[["ExeckWh", "IdlekWh"]] = df.apply(lambda x : calc_carbon(x.TotalExec, x.IdleSeconds, x.MemoryUsed, x.MemAllocated, x.NumCores), axis=1)
    df["TotalkWh"] = df.apply(lambda x : x.IdlekWh + x.ExeckWh, axis=1)
    static_alloc_kwh = df.TotalkWh.tolist()

    pct_reduction_static_to_dynamic_alloc = [100 * (static_alloc_kwh[i]-dynamic_alloc_kwh[i]) / static_alloc_kwh[i] for i in range(len(df))]

    plot_cdf(ax, pct_reduction_static_to_dynamic_alloc, label="Reduction in Emissions (%)", 
                forecaster="", filename="max_vs_avg_alloc", log=False, legend=False, ticklines=list(range(0,100,10)))

    plt.tight_layout()
    plt.close()


def workload_shifting(num_workers=14, slack=60):
    #exec_mem_cores_df = preproc_df()
    #df = add_transform_values(exec_mem_cores_df)
    #df.to_pickle("preproc.pickle")
    preproc_df = pd.read_pickle("preproc.pickle")
    slack_values = [2, 10, 30, 60]

    #for i, slack in enumerate(slack_values):
        #df = preproc_df
        #df.sort_values(by=["TotalInvocations"], ascending=False, inplace=True)
        #dfs = np.array_split(df, num_workers)
        
        #with ProcessPoolExecutor(max_workers = num_workers) as executor:
            #results = executor.map(multiproc_shift_workload, dfs, [slack] * num_workers) 

        #df = pd.concat(results, ignore_index=True)
        #df.to_pickle("shifted_{}.pickle".format(i))


    preproc_df["SumFractionalContainers"] = preproc_df.TransformedValues.apply(lambda x : sum(x))
    preproc_df["SumNoBatching"] = preproc_df.TransformedValues.apply(lambda x : sum(np.ceil(x)))

    optimal_sum = preproc_df.SumFractionalContainers.sum()
    no_batch_sum = preproc_df.SumNoBatching.sum()

    plt.figure(figsize=(5.8, 3.2), dpi=300)
    plt.ylabel("Reduction in Containers (%)")
    plt.xlabel("Relative Deadline (min)")

    reductions = []

    for i in range(4):
        df = pd.read_pickle("shifted_{}.pickle".format(i))
        df["SumShifted"] = df.ShiftedWorkload.apply(lambda x : sum(x))
        reductions.append(100 * (no_batch_sum - df.SumShifted.sum()) / no_batch_sum)
    
    plt.plot([0] + slack_values, [0] + reductions, marker="o", linestyle="--")
    plt.tight_layout()
    print("here")
    # put markers on datapoints
    plt.savefig(output_plots_dir + "workload_shifting.pdf")


def multiproc_shift_workload(df, slack):
    df["ShiftedWorkload"] = df.TransformedValues.apply(lambda x : shift_workload(x, slack=slack))

    return df


def shift_workload(avg_conc_per_min, slack=2):
    """Shift workloads up to the allocated slack time to batch requests into containers.
    input:
        avg_conc_per_min list[float]: Number/fraction of containers required to execute requests that arrived during a minute

    """
    total_queued = 0
    slack_q = deque()
    num_active_containers = 0
    shifted_traffic = np.zeros(len(avg_conc_per_min))

    for cur_min, num_containers in enumerate(avg_conc_per_min):
        total_queued += num_containers
        slack_q.append(num_containers)
        #print("Loop {}".format(cur_min))
        #print(num_active_containers, total_queued, slack_q)

        if total_queued < num_active_containers:
            num_active_containers = np.ceil(total_queued)
        # If there's a deadline this minute, must scale up resources as needed to meet it
        elif len(slack_q) == slack + 1 and slack_q[0] > num_active_containers:
            num_active_containers = np.ceil(slack_q[0])
        
        total_queued = max(total_queued - num_active_containers, 0)
        num_idle_containers = num_active_containers

        while slack_q and num_idle_containers >= 0:
            cur_queued = slack_q.popleft()
            
            if num_idle_containers >= cur_queued:
                num_idle_containers -= cur_queued
            else:
                cur_queued -= num_idle_containers
                slack_q.appendleft(cur_queued)
                break

        #print(num_active_containers, total_queued, slack_q)
        shifted_traffic[cur_min] = num_active_containers          

    return shifted_traffic


def group_minutes_across_apps(traffic_per_min, num_days):
    num_minutes = num_days * 1440
    agg_traffic_per_min = np.zeros(num_minutes)
    
    for traffic in traffic_per_min:
        agg_traffic_per_min = np.add(agg_traffic_per_min[:num_minutes], traffic[:num_minutes])

    return agg_traffic_per_min


def calc_carbon(exec_time, idle_time, used_mem, allocated_mem, num_cores):
    """ Compute kWh of energy consumed for used and wasted mem

    input:
        exec_time (int): in seconds
        idle_time (int): in seconds
        used_mem (list[int]): GB-s used during execution per block
        allocated_mem (list[int]): GB-s allocated per block
    """
    exec_mem_energy = sum(used_mem) * mem_power_draw
    idle_mem_energy = (sum(allocated_mem) - sum(used_mem)) * mem_power_draw
    exec_cpu_energy = exec_time * num_cores * utilized_cpu_power_draw

    return pd.Series([(exec_mem_energy + exec_cpu_energy) / SECONDS_IN_HOUR, (idle_mem_energy) / SECONDS_IN_HOUR]) 


def preproc_df(max_mem=False):
    hashapps = pd.read_pickle(cs_wm_forecaster_path.format("10_min_keepalive")).HashApp.tolist()
    exec_df = pd.read_pickle(exec_data_file)
    mem_df = pd.read_pickle(memory_data_file if max_mem == False else memory_data_file.replace("memory", "max_memory"))
    mem_df = mem_df[mem_df.HashApp.isin(hashapps)]
    df = mem_df.merge(exec_df, on="HashApp", how="left")

    df["NumCores"] = df.AverageMemUsage.apply(lambda x: np.ceil(np.max(x) / mb_per_core))
    df["TotalExec"] = df.TotalExec.apply(lambda x : x / 1000)

    return df

def plot_cdf(plot_obj, data, label, forecaster, filename, log=True, legend=True, ticklines=None):
    data = [round(val, 6) for val in data]
    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))

    if "10" in forecaster:
        linestyle = "--"
    elif "1" in forecaster:
        linestyle = "-."
    else:
        linestyle = "-"
    
    plot_obj.plot(cdfx, cdfy, label=forecaster, linestyle=linestyle)
    plot_obj.grid() 

    plot_obj.set_xlabel(label)
    plot_obj.set_ylabel("Fraction of Applications")
    if log:
        plot_obj.set_xscale("log")
    
    if legend:
        plot_obj.legend(fontsize=10)

    if ticklines:
        plot_obj.set_xticks(ticklines)

    plt.savefig(output_plots_dir + filename + ".pdf")


if __name__ == "__main__":
    forecasters = ["Default_Knative", "5_min_keepalive", "10_min_keepalive"]
    #plot_carbon_ratio_vs_cs_duration(forecasters)
    #plot_max_mem_vs_avg()
    #plot_layers()
    workload_shifting()