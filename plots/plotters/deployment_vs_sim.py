import pickle
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations
from pathlib import Path

sys.path.append("../../code")
from results.utils import add_transform_values
from knative_deployment.full_workload import parse_full_workload

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

hashapp_list_path = data_dir + "train_test_split/{}_{}_apps.pickle"
deployment_data_dir = data_dir + "knative_deployment_data/"
cold_start_save_path = deployment_data_dir + "{}_sim_result.pickle"
sample_data = data_dir + "knative_deployment_data/workload_invocation_data_max_{}_{}_apps.pickle"

MIN_METRIC = 1e-6
COLD_START_DURATION = 0.808
COLD_START_TIME_WEIGHT = 1
WASTED_MEMORY_WEIGHT = 1 / 99.69

RESULT_COLS = ["NumColdStarts", "MemAllocated", "MemoryUsed"]


def gen_results(forecasters):
    df = pd.read_pickle(deployment_data_dir + "/workload_invocation_data_max_63_100_apps.pickle")

    start_minute = 5
    cutoff = 23*60 + 55
    print(df)
    #df.NumInvocations = df.InvocationsPerMin.apply(lambda x : sum(x[:cutoff]))
    
    #print("Num requests: {}".format(df.NumInvocations.sum()))
    for forecaster in forecasters:
        df = pd.read_pickle(cold_start_save_path.format(forecaster))
        print("for {}".format(forecaster))
        df.NumColdStarts = df.NumColdStarts.apply(lambda x : sum(x[start_minute:cutoff]))
        df.MemAllocated = df.MemAllocated.apply(lambda x : sum(x[start_minute:cutoff]))
        df.MemoryUsed = df.MemoryUsed.apply(lambda x : sum(x[start_minute:cutoff]))
        
        print("Mem alloc {}".format(df.MemAllocated.sum()))
        print("Mem used {}".format(df.MemoryUsed.sum()))
        print("Num cold starts {}".format(df.NumColdStarts.sum()))
        print("Wasted mem {}".format(df.MemAllocated.sum() - df.MemoryUsed.sum()))
        weighted_mem = (df.MemAllocated.sum() - df.MemoryUsed.sum()) * WASTED_MEMORY_WEIGHT
        weighted_cs = df.NumColdStarts.sum() * COLD_START_DURATION
        RUM = weighted_mem + weighted_cs
        print("RUM {}".format(RUM))


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


def powerset(features):
        return list(chain.from_iterable(combinations(features, r) for r in range(len(features)+1)))[1:]


def gen_comparison_plot(forecaster):
    cutoff = 10*60    
    cs_df = pd.read_pickle(cold_start_save_path.format(forecaster))
    df = pd.read_pickle(deployment_data_dir + "/workload_invocation_data_max_63_100_apps.pickle")    
    df = add_transform_values(df)

    plt.subplot(4,1,1)

    df["ContainersPerMin"] = df.TransformedValues.apply(lambda x : np.ceil(x))
    df.ContainersPerMin = df.ContainersPerMin.apply(lambda x : x[:cutoff])

    containers_per_min_per_app = np.array(df.ContainersPerMin.values.tolist())
    agg_containers_per_min = [sum(containers_per_min_per_app[:, cur_min]) for cur_min in range(cutoff)]
    plt.plot(list(range(cutoff)), agg_containers_per_min)

    plt.ylabel("Pods (Load Gen)")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("../output_plots/azure/sim_vs_deployment.pdf")


    plt.subplot(4,1,2)

    cs_per_min_per_app = np.array(cs_df.NumColdStarts.values.tolist())
    agg_cs_per_min = [sum(cs_per_min_per_app[:, cur_min]) for cur_min in range(cutoff)]
    plt.plot(list(range(cutoff)), agg_cs_per_min)

    plt.ylabel("Cold Starts")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("../output_plots/azure/sim_vs_deployment.pdf")

    plt.subplot(4,1,3)

    df.TransformedValues = df.TransformedValues.apply(lambda x : x[:cutoff])
    avg_conc_per_min_per_app = np.array(df.TransformedValues.values.tolist())
    avg_conc_per_min_per_app = [sum(avg_conc_per_min_per_app[:,cur_min]) for cur_min in range(cutoff)]
    plt.plot(list(range(cutoff)), avg_conc_per_min_per_app)

    plt.ylabel("Avg Req Conc")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("../output_plots/azure/sim_vs_deployment.pdf")

    plt.subplot(4,1,4)
    df = pd.read_pickle(deployment_data_dir + "/workload_invocation_data_max_63_100_apps.pickle")    
    df = parse_full_workload(df.HashApp.tolist(), group=False)

    df.InvocationsPerMin = df.InvocationsPerMin.apply(lambda x : x[:cutoff])
    inv_per_min_per_app = np.array(df.InvocationsPerMin.values.tolist())
    agg_inv_per_min = [sum(inv_per_min_per_app[:,cur_min]) for cur_min in range(cutoff)]
    plt.plot(list(range(cutoff)), agg_inv_per_min)

    plt.ylabel("Invocations")
    plt.ylim(bottom=0)
    plt.xlabel("Time (min)")
    plt.tight_layout()
    plt.savefig("../output_plots/azure/sim_vs_deployment.pdf")


if __name__ == '__main__':
    forecasters = ["femux", "Default_Knative"]

    gen_results(forecasters)
    gen_comparison_plot("femux")
