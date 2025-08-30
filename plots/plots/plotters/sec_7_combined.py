import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

sys.path.append("../../")
from plotter import plot

from pathlib import Path

plt.rcParams["pdf.fonttype"] = 42

fig, axs = plt.subplots(1, 4, figsize=(13, 3.75), dpi=300)
fig.subplots_adjust(wspace=0)
# change the width proportion of the subplots
axs[0].set_box_aspect(1)
axs[1].set_box_aspect(1)
axs[2].set_box_aspect(1.618)
# axs[3].set_box_aspect(1)

data_dir = str(Path(__file__).resolve().parents[2] / "data") + "/"

sample_data = (
    data_dir + "knative_deployment_data/workload_invocation_data_max_63_100_apps.pickle"
)
full_workload = data_dir + "knative_deployment_data/full_workload_parsed_data.pickle"

deployment_data_dir = data_dir + "knative_deployment_data/"
sim_cold_start_save_path = deployment_data_dir + "{}_sim_result.pickle"
deployment_mem_data = deployment_data_dir + "{}_per_app_alloc_util.pickle"
deployment_cs_data = deployment_data_dir + "{}_per_app_cold_starts.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

RESULT_COLS = ["NumColdStarts", "MemAllocated", "MemoryUsed"]


def get_hashfunc(app_name, hashfuncs):
    for hashfunc in hashfuncs:
        if hashfunc[:8] == app_name[-8:]:
            return hashfunc

    raise Exception("Func missing?")


def get_RUM(cs, mem_alloc, mem_util):
    return cs * COLD_START_DURATION + (mem_alloc - mem_util) * WASTED_MEMORY_WEIGHT


def preproc_deployment_data(hashfuncs):
    dfs = []
    forecasters = ["vanilla", "femux"]

    for forecaster in forecasters:
        mem_df = pd.read_pickle(deployment_mem_data.format(forecaster))
        cs_df = pd.read_pickle(deployment_cs_data.format(forecaster))

        print("# failed reqs", cs_df.FAILED_REQUEST_COUNT.sum())

        df = mem_df.merge(cs_df, on="APP_NAME", how="right")
        df.MEM_ALLOC = df.MEM_ALLOC.apply(lambda x: x / 1000)
        df.MEM_UTIL = df.MEM_UTIL.apply(lambda x: x / 1000)

        df = df[df.REQUEST_COUNT > 0]

        print("Deployment req count {}".format(df.REQUEST_COUNT.sum()))
        df["HashApp"] = df.APP_NAME.apply(lambda x: get_hashfunc(x, hashfuncs))
        df["RUMDeploy"] = df.apply(
            lambda x: get_RUM(x.COLD_START_COUNT, x.MEM_ALLOC, x.MEM_UTIL), axis=1
        )
        df["CSPctDeploy"] = df.apply(
            lambda x: (x.COLD_START_COUNT / x.REQUEST_COUNT) * 100, axis=1
        )

        dfs.append(df)

    # get the % decrease in RUM, multiply by -100 to get % reduction as positive number
    df = dfs[0].merge(dfs[1], on="HashApp")
    print(
        "Total rum deployment: knative {}, femux {}".format(
            df.RUMDeploy_x.sum(), df.RUMDeploy_y.sum()
        )
    )
    print(
        "Mem Wasted and cold starts: knative {}GB-s/{}cold start, femux {}GB-s/{}cold start".format(
            df.MEM_ALLOC_x.sum() - df.MEM_UTIL_x.sum(),
            df.COLD_START_COUNT_x.sum(),
            df.MEM_ALLOC_y.sum() - df.MEM_UTIL_y.sum(),
            df.COLD_START_COUNT_y.sum(),
        )
    )
    return df


def gen_results():
    inv_df = pd.read_pickle(
        deployment_data_dir + "workload_invocation_data_max_63_100_apps.pickle"
    )
    deployment_df = preproc_deployment_data(inv_df.HashApp.tolist())

    start_minute = 5
    cutoff = 23 * 60 + 55

    forecasters = ["Default_Knative", "femux"]
    dfs = []

    for forecaster in forecasters:
        df = pd.read_pickle(sim_cold_start_save_path.format(forecaster))
        df = df.merge(inv_df, on="HashApp")

        df.NumColdStarts = df.NumColdStarts.apply(lambda x: sum(x[start_minute:cutoff]))
        df.MemAllocated = df.MemAllocated.apply(lambda x: sum(x[start_minute:cutoff]))
        df.MemoryUsed = df.MemoryUsed.apply(lambda x: sum(x[start_minute:cutoff]))

        df["RUMSim"] = df.apply(
            lambda x: get_RUM(x.NumColdStarts, x.MemAllocated, x.MemoryUsed), axis=1
        )
        dfs.append(df)

    ## get the % decrease in RUM, multiply by -100 to get % reduction as positive number
    sim_df = dfs[0].merge(dfs[1], on="HashApp")

    print("Total rum {} sim: {}".format("femux", sim_df.RUMSim_y.sum()))
    print(
        "CS {} and MemWasted {} GB-s".format(
            sim_df.NumColdStarts_y.sum(),
            sim_df.MemAllocated_y.sum() - sim_df.MemoryUsed_y.sum(),
        )
    )
    print("Total rum {} sim: {}".format("vanilla", sim_df.RUMSim_x.sum()))
    print(
        "CS {} and MemWasted {} GB-s".format(
            sim_df.NumColdStarts_x.sum(),
            sim_df.MemAllocated_x.sum() - sim_df.MemoryUsed_x.sum(),
        )
    )

    plot_line(deployment_df.CSPctDeploy_x.tolist(), "Knative")
    plot_line(deployment_df.CSPctDeploy_y.tolist(), "FeMux")

    raw_rums = {"Evaluation Method": [], "Baseline": [], "Default RUM": []}
    raw_rums["Evaluation Method"].append("Deployment")
    raw_rums["Baseline"].append("Knative")
    raw_rums["Default RUM"].append(deployment_df.RUMDeploy_x.sum())
    raw_rums["Evaluation Method"].append("Deployment")
    raw_rums["Baseline"].append("FeMux")
    raw_rums["Default RUM"].append(deployment_df.RUMDeploy_y.sum())

    raw_rums_df = pd.DataFrame(raw_rums)

    sns.barplot(
        x="Evaluation Method",
        y="Default RUM",
        hue="Baseline",
        data=raw_rums_df,
        ax=axs[2],
        legend=False,
    )
    
    axs[2].set_ylabel("RUM")
    # axs[2].yaxis.set_label_coords(-0.2, 0.5)
    axs[2].set_xlabel("")
    # set x-axis ticks as "Knative" and "FeMux"
    axs[2].set_xticks([-0.2, 0.2])
    axs[2].set_xticklabels(["Knative", "FeMux"], rotation=-15)


def plot_line(data, label):
    if label == "FeMux":
        linestyle = "--"
    else:
        linestyle = "-"

    data = [round(val, 6) for val in data]
    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))
    # plt.set_cmap("cividis")
    axs[1].plot(
        cdfx, cdfy, label=label, linestyle=linestyle
    )
    axs[1].grid(alpha=0.2)
    axs[1].set_xlabel("Cold Start (%)")
    axs[1].set_ylabel("Fraction of Applications")
    axs[1].legend()


def plot_inv_per_func():
    names = ["Azure Trace", "Eval Subtrace"]
    for i, name in enumerate([full_workload, sample_data]):
        data = pd.read_pickle(name)

        print(data)

        data = data.NumInvocations.tolist()

        data = [round(val, 6) for val in data]
        cdfx = np.sort(data)
        cdfy = np.linspace(1 / len(data), 1.0, len(data))
        linestyle = "-" if i % 2 == 0 else "--"
        axs[0].plot(cdfx, cdfy, label=names[i], linestyle=linestyle)

        axs[0].set_xscale("log")

        axs[0].set_xlabel("Number of Invocations")
        axs[0].set_ylabel("Fraction of Applications")
        axs[0].grid(alpha=0.2)
        axs[0].legend()


def plot_scalability():
    weight_map = {"H": 1, "M": 534, "1": 15, "5": 393, "E": 8, "F": 2, "A": 22, "S": 22}
    forecasters = ["5", "10", "holt", "mc", "es", "fft", "ar", "setar"]

    df = pd.DataFrame()
    for i in range(len(forecasters)):
        df = pd.concat(
            [
                df,
                pd.read_pickle(
                    f"../../data/knative_scalability_data/per_forecaster_data_v2/{forecasters[i]}_horizontal_scalability_data.pickle"
                ),
            ]
        )

    # inference latency
    latency_map = {}
    buckets = [20, 40, 60, 80, 100]
    median_latency = {}
    mean_latency = {}
    nn_latency = {}
    median_latency_ci = {}
    nn_latency_ci = {}

    for _, row in df.iterrows():
        # for bucket in buckets:
        apps = row["FORECASTER"][0] + "_" + str(row["RPS"])
        if apps not in latency_map:
            latency_map[apps] = []
        if "1" == apps[0] or "5" == apps[0]:
            latency_map[apps].extend([0] * len(row["RESPONSE_TIMES"][-1000:]))
        else:
            latency_map[apps].extend(row["RESPONSE_TIMES"][-1000:])
    for key, val in latency_map.items():
        latency_map[key] = val * weight_map[key.split("_")[0]]
        if "F" in key:
            print(len(latency_map[key]))

    cur_uci_med = []
    cur_lci_med = []
    cur_resp_med = []

    cur_uci_p95 = []
    cur_lci_p95 = []
    cur_resp_p95 = []
    overall_median_latency = {}
    overall_nn_latency = {}

    for bucket in buckets:
        same_bucket_list = []
        for key, val in latency_map.items():
            if f"{bucket}" in key:
                same_bucket_list += val

        # shuffle same_bucket_list to avoid any ordering bias and again get the list
        same_bucket_list.sort()
        same_bucket_list = np.random.permutation(same_bucket_list).tolist()

        chunked_p95_times = []
        chunked_med_times = []

        chunk_size = 1000

        for i in range(0, len(same_bucket_list), chunk_size):
            chunked_p95_times.append(
                np.percentile(same_bucket_list[i : i + chunk_size], 99)
            )
            chunked_med_times.append(np.median(same_bucket_list[i : i + chunk_size]))

        print(len(chunked_med_times), len(chunked_p95_times))
        median = np.mean(chunked_med_times)
        median_latency[bucket] = median
        overall_median_latency[bucket] = np.mean(same_bucket_list)

        # mean_latency[bucket] = np.mean(same_bucket_list)
        nn = np.percentile([elem for elem in chunked_p95_times if elem != None], 99)
        nn_latency[bucket] = nn
        overall_nn_latency[bucket] = np.percentile(same_bucket_list, 99)

        # Calculate the standard error of the mean
        sem_p95 = stats.sem(chunked_p95_times)
        sem_med = stats.sem(chunked_med_times)

        # Calculate the confidence intervals
        ci_med = stats.t.interval(
            0.95,
            len(chunked_med_times) - 1,
            loc=np.median(same_bucket_list),
            scale=sem_med,
        )
        median_latency_ci[bucket] = ci_med
        cur_uci_med.append(ci_med[1] * 1000)
        cur_lci_med.append(ci_med[0] * 1000)
        cur_resp_med.append(median * 1000)

        ci_nn = stats.t.interval(
            0.95,
            len(chunked_p95_times) - 1,
            loc=np.percentile(same_bucket_list, 99),
            scale=sem_p95,
        )
        nn_latency_ci[bucket] = ci_nn
        cur_uci_p95.append(ci_nn[1] * 1000)
        cur_lci_p95.append(ci_nn[0] * 1000)
        cur_resp_p95.append(nn * 1000)

    # Add error bars to the plot
    axs[3].errorbar(
        buckets,
        [overall_median_latency[bucket] * 1000 for bucket in buckets],
        yerr=[((upper - lower) * 1000) for lower, upper in median_latency_ci.values()],
        marker=".",
        capsize=5,
        color="blue",
        label="mean",
    )

    axs[3].errorbar(
        buckets,
        [overall_nn_latency[bucket] * 1000 for bucket in buckets],
        yerr=[((upper - lower) * 1000) for lower, upper in nn_latency_ci.values()],
        marker=".",
        capsize=5,
        color="orange",
        label="p99",
    )

    axs[3].set_xlabel("Forecasting Rate (rps)")
    axs[3].set_ylabel("Forecasting Latency (ms)")

    axs[3].set_xticks(buckets)
    axs[3].set_xticklabels(buckets)

    # create a second x-axis at the top of the plot
    ax2 = axs[3].twiny()

    # calculate the new x-values for the top x-axis
    new_xticks = [value * 60 for value in buckets]
    new_xticklabels = [value * 60/1000 for value in buckets]

    # set the xticks and xticklabels for the top x-axis
    ax2.set_xticks(new_xticks)
    ax2.set_xticklabels(new_xticklabels)
    ax2.set_xlabel("Number of Apps (x1000)")

    # calculate the padding for the x-axis limits
    padding = (max(buckets) - min(buckets)) * 0.05

    # set the limits of both axes to be the same with padding
    axs[3].set_xlim([min(buckets) - padding, max(buckets) + padding])
    ax2.set_xlim([min(new_xticks) - padding * 60, max(new_xticks) + padding * 60])
    axs[3].legend()

    axs[3].set_ylim([0, 30])

    # add grid
    axs[3].grid(alpha=0.2)
    ax2.grid(alpha=0.2)


if __name__ == "__main__":
    gen_results()
    plot_inv_per_func()
    plot_scalability()

    plt.tight_layout()
    plt.savefig("../output_plots/sec_7_combined.pdf")
