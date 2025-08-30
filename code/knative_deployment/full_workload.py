import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append("..")
from transform.utils import gen_events
from results.utils import init_df

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

invocation_data = data_dir + "preproc_data/invocation_data/preprocessed_data_{:02d}.pickle"

parsed_data_path = data_dir + "knative_deployment_data/{}_workload_parsed_data.pickle"

NUM_INV_FILES = 40
MINUTES_PER_DAY = 1440

def parse_full_workload(hashapps, group=True):
    dfs = []

    for filenum in range(NUM_INV_FILES):
        invocation_df = pd.read_pickle(invocation_data.format(filenum))

        # parse only first day of data
        invocation_df.InvocationsPerMin = invocation_df.InvocationsPerMin.apply(lambda x : x[:MINUTES_PER_DAY])
        invocation_df.ExecDurations = invocation_df.ExecDurations.apply(lambda x : x[0])
        invocation_df.NumInvocations = invocation_df.InvocationsPerMin.apply(lambda x : sum(x))
        
        if hashapps:
            invocation_df = invocation_df[invocation_df.HashApp.isin(hashapps)]

        if group:
            invocation_df = invocation_df.groupby("HashApp").agg({
                "NumInvocations": "sum"
            }).reset_index()

        invocation_df = invocation_df[invocation_df.NumInvocations > 0]

        dfs.append(invocation_df)

    parsed_df = pd.concat(dfs, ignore_index=True)

    return parsed_df

def avg_exec_time_per_app(inv_df):
    """Get the parsed invocation dataframe (output of above function) which includes HashApp, HashFunc, and ExecDurations.
    Group all the Hashfuncs per app and get the average execution time based on the number of functions.
    """
    exec_df = inv_df.groupby("HashApp").agg({
                "HashFunction": list,
                "ExecDurations": "sum",
            }).reset_index()

    # Average the sum of execution times across functions by the number of functions.
    exec_df["ExecDurations"] = exec_df.apply(lambda x : x.ExecDurations / len(x.HashFunction), axis=1)
    
    return exec_df



if __name__ == "__main__":
    for split in ["full", "test", "train"]:
        hashapps = None if split == "full" else init_df(split, 100).HashApp.tolist()
        parsed_df = parse_full_workload(hashapps)
        parsed_df.to_pickle(parsed_data_path.format(split))