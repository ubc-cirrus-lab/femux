import pandas as pd
import numpy as np
from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"

preproc_path = data_dir + "preproc_data/invocation_data/preprocessed_data_{:02d}.pickle"
save_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"

MINUTES_PER_DAY = 1440


def get_invocations(num_days=12):
    print("Adding invocation traces")
    invocation_dfs = []
    for filenum in range(40):
        inv_df = pd.read_pickle(preproc_path.format(filenum))
        
        inv_df = inv_df[["HashApp", "InvocationsPerMin", "ExecDurations"]]

        invocation_dfs.append(inv_df)
    
    invocation_df = pd.concat(invocation_dfs, ignore_index=True)

    invocation_df["TotalInvocations"] = invocation_df.InvocationsPerMin.apply(lambda x: sum(x[:(num_days) * MINUTES_PER_DAY])) 
    invocation_df["TotalExec"] = invocation_df.apply(lambda x : total_exec_time(x.InvocationsPerMin, x.ExecDurations, num_days), axis=1)

    invocation_df = invocation_df.groupby("HashApp").agg({"TotalInvocations": sum, "TotalExec": sum, "InvocationsPerMin": list})
    invocation_df = invocation_df.reset_index()

    invocation_df["InvocationsPerMin"] = invocation_df.InvocationsPerMin.apply(lambda x: group_inv_count(x, num_days))

    print(invocation_df)
    invocation_df.to_pickle(save_path.format(num_days))


def total_exec_time(invocations_per_min, exec_durations, num_days): 
    total_exec_time = 0
    
    for cur_day in range(num_days):
        exec_duration = exec_durations[cur_day]
        total_exec_time += sum(invocations_per_min[cur_day * MINUTES_PER_DAY: (cur_day + 1) * MINUTES_PER_DAY]) * exec_duration

    return total_exec_time


def group_inv_count(func_invocation_list, num_days):
    num_minutes = num_days * MINUTES_PER_DAY
    app_inv_counts = np.zeros(num_minutes)
    
    for func_invocations in func_invocation_list:
        app_inv_counts = np.add(app_inv_counts[:num_minutes], func_invocations[:num_minutes])

    return app_inv_counts

if __name__ == "__main__":
    get_invocations(12)