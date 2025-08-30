import pandas as pd
import numpy as np

data_path = "../../data/azure/preproc_data/invocation_data/preprocessed_data_00.pickle"
save_path = "../../data/azure/preproc_data/app_exec_time_data.pickle"

NUM_PREPROC_FILES = 40
NUM_DAYS = 14

def gen_exec():
    """ generate app-level execution times by taking the average daily execution time
    across all functions within an app

    returns: pd.dataframe
    columns:
        HashApp: str
            Application hash
        AverageMemUsage: list[int]
            execution time per day in ms
    """
    dfs = []

    for cur_day in range(NUM_PREPROC_FILES):
        print("getting app execution data for file {}".format(cur_day))
        cur_df = pd.read_pickle(data_path.replace("00", "{:02d}".format(cur_day)))
        dfs.append(cur_df[["HashApp", "HashFunction", "ExecDurations"]])
    
    df = pd.concat(dfs, ignore_index=True)

    df = df.groupby("HashApp").agg({"ExecDurations": list})

    df["ExecDurations"] = df.ExecDurations.apply(lambda x : average_exec_times(x))
    print(df)

    df.to_pickle(save_path)


def average_exec_times(exec_time_lists):
    avg_list = []

    for cur_day in range(NUM_DAYS):
        cur_sum = 0
        num_execs = 0
        
        for exec_time_list in exec_time_lists:
            cur_exec = exec_time_list[cur_day]
            
            if cur_exec > 0:
                num_execs += 1
                cur_sum += cur_exec
        
        avg_list.append(int(cur_sum / max(num_execs, 1)))

    print(avg_list)
    return np.array(avg_list)


if __name__ == '__main__':
    gen_exec()



