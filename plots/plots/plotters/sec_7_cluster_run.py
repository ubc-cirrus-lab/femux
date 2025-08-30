import pickle
import pandas as pd
import numpy as np
from itertools import chain, combinations
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42

axs = plt.figure(figsize=(5.64, 3)).subplots(1, 2)
ax = axs[0]
ax2 = axs[1]

data_dir = str(Path(__file__).resolve().parents[2] / "data") + "/"
save_path = str(Path(__file__).parents[1] / "output_plots") + "/"

deployment_data_dir = data_dir + "knative_deployment_data/"
sim_cold_start_save_path = deployment_data_dir + "{}_sim_result.pickle"
deployment_mem_data = deployment_data_dir + "{}_per_app_alloc_util.pickle"
deployment_cs_data = deployment_data_dir + "{}_per_app_cold_starts.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

RESULT_COLS = ["NumColdStarts", "MemAllocated", "MemoryUsed"]


def preproc_deployment_data(hashfuncs):
    dfs = []
    forecasters = ["vanilla", "femux"]
    
    for forecaster in forecasters:
        mem_df = pd.read_pickle(deployment_mem_data.format(forecaster))
        cs_df = pd.read_pickle(deployment_cs_data.format(forecaster))
        
        print("# failed reqs", cs_df.FAILED_REQUEST_COUNT.sum())

        df = mem_df.merge(cs_df, on="APP_NAME", how="right")
        df.MEM_ALLOC = df.MEM_ALLOC.apply(lambda x : x / 1000)
        df.MEM_UTIL = df.MEM_UTIL.apply(lambda x : x / 1000)

        df = df[df.REQUEST_COUNT > 0]

        print("Deployment req count {}".format(df.REQUEST_COUNT.sum()))
        df["HashApp"] = df.APP_NAME.apply(lambda x : get_hashfunc(x, hashfuncs))
        df["RUMDeploy"] = df.apply(lambda x : get_RUM(x.COLD_START_COUNT, x.MEM_ALLOC, x.MEM_UTIL) ,axis=1)
        df["CSPctDeploy"] = df.apply(lambda x : (x.COLD_START_COUNT / x.REQUEST_COUNT) * 100,axis=1)
        
        dfs.append(df)
        
    
    # get the % decrease in RUM, multiply by -100 to get % reduction as positive number
    df = dfs[0].merge(dfs[1], on="HashApp")
    print("Total rum deployment: knative {}, femux {}".format(df.RUMDeploy_x.sum(), df.RUMDeploy_y.sum()))
    print("Mem Wasted and cold starts: knative {}GB-s/{}cold start, femux {}GB-s/{}cold start".format(df.MEM_ALLOC_x.sum() - df.MEM_UTIL_x.sum(), 
                                                                                                        df.COLD_START_COUNT_x.sum(), df.MEM_ALLOC_y.sum() - df.MEM_UTIL_y.sum(),
                                                                                                        df.COLD_START_COUNT_y.sum()))

    return df

def gen_results():
    inv_df = pd.read_pickle(deployment_data_dir + "workload_invocation_data_max_63_100_apps.pickle")
    deployment_df = preproc_deployment_data(inv_df.HashApp.tolist())

    start_minute = 5
    cutoff = 23 * 60 + 55
    
    forecasters = ["Default_Knative", "femux"]
    dfs = []

    for forecaster in forecasters:
        df = pd.read_pickle(sim_cold_start_save_path.format(forecaster))
        df = df.merge(inv_df, on="HashApp")
        
        df.NumColdStarts = df.NumColdStarts.apply(lambda x : sum(x[start_minute:cutoff]))
        df.MemAllocated = df.MemAllocated.apply(lambda x : sum(x[start_minute:cutoff]))
        df.MemoryUsed = df.MemoryUsed.apply(lambda x : sum(x[start_minute:cutoff]))
        
        df["RUMSim"] = df.apply(lambda x : get_RUM(x.NumColdStarts, x.MemAllocated, x.MemoryUsed) ,axis=1)
        dfs.append(df)
 
    ## get the % decrease in RUM, multiply by -100 to get % reduction as positive number
    sim_df = dfs[0].merge(dfs[1], on="HashApp")

    print("Total rum {} sim: {}".format("femux", sim_df.RUMSim_y.sum()))
    print("CS {} and MemWasted {} GB-s".format(sim_df.NumColdStarts_y.sum(), sim_df.MemAllocated_y.sum() - sim_df.MemoryUsed_y.sum()))
    print("Total rum {} sim: {}".format("vanilla", sim_df.RUMSim_x.sum()))
    print("CS {} and MemWasted {} GB-s".format(sim_df.NumColdStarts_x.sum(), sim_df.MemAllocated_x.sum() - sim_df.MemoryUsed_x.sum()))

    plot_line(deployment_df.CSPctDeploy_x.tolist(), "Knative")
    plot_line(deployment_df.CSPctDeploy_y.tolist(), "FeMux")
    
    raw_rums = {"Evaluation Method":[], "Baseline": [], "Default RUM":[]}
    raw_rums["Evaluation Method"].append("Deployment")
    raw_rums["Baseline"].append("Knative")
    raw_rums["Default RUM"].append(deployment_df.RUMDeploy_x.sum())
    raw_rums["Evaluation Method"].append("Deployment")
    raw_rums["Baseline"].append("FeMux")
    raw_rums["Default RUM"].append(deployment_df.RUMDeploy_y.sum())
    
    raw_rums_df = pd.DataFrame(raw_rums)
    
    sns.barplot(x="Evaluation Method", y="Default RUM", hue="Baseline", data=raw_rums_df, ax=ax2, legend=False)   
    

def get_RUM(cs, mem_alloc, mem_util):
    return cs * COLD_START_DURATION + (mem_alloc - mem_util) * WASTED_MEMORY_WEIGHT


def get_hashfunc(app_name, hashfuncs):
    for hashfunc in hashfuncs:
        if hashfunc[:8] == app_name[-8:]:
            return hashfunc
    
    raise Exception("Func missing?")
 

def plot_line(data, label):
    if label == "FeMux":
        linestyle = "--"
    else:
        linestyle = "-"

    data = [round(val, 6) for val in data]
    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))
    # plt.set_cmap("cividis")
    ax.plot(cdfx, cdfy, label=label, linestyle=linestyle)
        
    ax.set_xlabel("Cold Start (%)")
    ax.set_ylabel("Fraction of Applications")
    ax.legend()


if __name__ == '__main__':
    gen_results()
    
    # vertical line at x= 0
    #ax.axvline(x=1, color="black", linestyle="--", linewidth=0.5)
    # log x axis
    #ax.set_xscale("log")
    ax.grid(alpha=0.2)
    # remove x axis label
    ax2.set_xlabel("")
    
    # scientific notation for y axis of ax2
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    
    plt.tight_layout()
    plt.savefig(save_path + "sec_7_cluster_run.pdf")