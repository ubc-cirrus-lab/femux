import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from full_workload import parse_full_workload, avg_exec_time_per_app

sys.path.append("..")
from transform.concurrency_transformer import ConcurrencyTransformer
from results.utils import init_df, add_mem_values

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

invocation_data = data_dir + "preproc_data/invocation_data/preprocessed_data_{:02d}.pickle"
transformed_data = data_dir + "transformed_data/concurrency/app/small_app_conc_{:02d}.pickle"

parsed_singlefunc_path = data_dir + "knative_deployment_data/parsed_singlefunc_data.pickle"
parsed_app_data_path = data_dir + "knative_deployment_data/parsed_app_data.pickle"
workload_path = data_dir + "knative_deployment_data/workload_invocation_data_{}.pickle"
workload_event_path = data_dir + "knative_deployment_data/workload_event_data_{}.pickle"

NUM_FILES = 40
MINUTES_PER_DAY = 1440
# we want max 80% peak util, and have capacity for ~90 pods
MAX_NUM_CONTAINERS = 63
MIN_TOTAL_TRAFFIC = 48000
NUM_APPS = 100
# go for >200 apps, and add 2 variants to 100

def parse_container_conc_data():
    test_hashapps = init_df("test", 100).HashApp.tolist()

    dfs = []

    for filenum in range(NUM_FILES):
        df = pd.read_pickle(transformed_data.format(filenum))
        df = df[df.HashApp.isin(test_hashapps)]

        # parse only first day of data
        df["ContainersPerMin"] = df.TransformedValues.apply(lambda x : np.ceil(x))
        df.ContainersPerMin = df.ContainersPerMin.apply(lambda x : x[:MINUTES_PER_DAY])
        df.drop(columns=["ContainerInvocationsPerMin", "TransformedValues"], inplace=True)
        
        df["MaxContainerMinute"] = df.ContainersPerMin.apply(lambda x : max(x))
        df = df[df.NumEvents > 0]
        df = df[df.MaxContainerMinute < MAX_NUM_CONTAINERS]

        dfs.append(df)

    parsed_df = pd.concat(dfs, ignore_index=True)
    
    # add in memory values and remove ones with 0 memory usage
    parsed_df = add_mem_values(parsed_df)
    parsed_df.dropna(inplace=True)
    parsed_df.AverageMemUsage = parsed_df.AverageMemUsage.apply(lambda x : x[0])
    parsed_df = parsed_df[parsed_df.AverageMemUsage > 0]

    print(parsed_df)

    inv_df = parse_full_workload(parsed_df.HashApp.tolist())
 
    parsed_df.merge(inv_df, on="HashApp").to_pickle(parsed_app_data_path)
    print("saved parsed df")


def gen_workload(parsed_path, workload_filename):
    """
    Generate a workload using the specified max containers per minute across apps, which 
    also has a minimum total traffic threshold.
    """
    parsed_df = pd.read_pickle(parsed_path)
    sampled_df = parsed_df.sample(n=NUM_APPS)

    while not check_valid(sampled_df):
        sampled_df = parsed_df.sample(n=NUM_APPS)

    sampled_df.to_pickle(workload_path.format(workload_filename))


def check_valid(df):
    """ Check that max number of containers in a minute across apps is below threshold, and 
    that the total number of invocations is above a threshold.
    """
    if df.NumInvocations.sum() < MIN_TOTAL_TRAFFIC:
        return False

    container_count_lists = df.ContainersPerMin.tolist()
    total_inv_counts = np.zeros(len(container_count_lists[0]))
    
    for container_count_list in container_count_lists:
        total_inv_counts = np.add(total_inv_counts, container_count_list)
    
    print(max(total_inv_counts))

    return max(total_inv_counts) < MAX_NUM_CONTAINERS


def gen_workload_events(workload_filename):
    """ Need IATs for FaasProfiler.
    """
    df = pd.read_pickle(workload_path.format(workload_filename))
    print("original df")
    print(df)
 
    inv_df = parse_full_workload(df.HashApp.tolist(), group=False)
 
    exec_df = avg_exec_time_per_app(inv_df)
    df = df.merge(exec_df, on="HashApp")
    df = df[["HashApp", "ExecDurations", "AverageMemUsage"]]

    transformer = ConcurrencyTransformer()
    inv_df["ExecDurations"] = inv_df.ExecDurations.apply(lambda x: [int(x)])
    inv_df = transformer.transform_concurrency_events(inv_df, MINUTES_PER_DAY, func_mode=False)

    print(inv_df[["HashApp", "ConcurrencyEvents"]])

    inv_df["IAT"] = inv_df.ConcurrencyEvents.apply(lambda x : gen_iats(x))

    inv_df = inv_df[["HashApp", "IAT"]]
    print(inv_df[["HashApp", "IAT"]])
    
    df = df.merge(inv_df, on="HashApp")
    df = df[["HashApp", "IAT", "ExecDurations", "AverageMemUsage"]]

    print(df)
    df.to_pickle(workload_event_path.format(workload_filename))


def gen_iats(concurrency_events):
    """Given a list of concurrency events which are tuples of (time, concurrency), generate a list 
    of inter-arrival times (IATs)
    
    returns: np.array
        list of IATs in seconds with the first IAT being the first arrival time.

    """
    print(concurrency_events[:5])
    # Get arrival times which are concurrency events where the previous event has a lower concurrency
    arrival_times = [concurrency_events[i][0] for i in range(1, len(concurrency_events)) if concurrency_events[i-1][1] < concurrency_events[i][1]]
    arrival_times = [concurrency_events[0][0]] + arrival_times
    print(arrival_times[:5])

    iat = [arrival_times[0]]
    for i in range(len(arrival_times)-1):
        iat.append(arrival_times[i+1] - arrival_times[i])

    return np.array(iat)


if __name__ == "__main__":
    workload_filename = "max_{}_{}_apps".format(MAX_NUM_CONTAINERS, NUM_APPS) 
    #parse_container_conc_data()
    #gen_workload(parsed_app_data_path, workload_filename)
    gen_workload_events(workload_filename)