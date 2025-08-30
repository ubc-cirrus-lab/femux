import pandas as pd
import numpy as np

data_path = "../../data/azure/azure_data/app_memory_percentiles.anon.d01.csv"
save_path = "../../data/azure/preproc_data/memory_data.pickle"

NUM_MEMORY_FILES = 12

def gen_memory():
    """ combine the memory usage for each application across all 12 days in a list.
    Memory values are in MB/minute

    returns: pd.dataframe
    columns:
        HashApp: str
            Application hash
        AverageMemUsage: list[int]
            memory usage in MB for each day (0 if not executed that day)
    """
    memory_df = pd.read_csv(data_path)

    memory_df["AverageMemUsage"] = memory_df.AverageAllocatedMb.apply(lambda x : [x])
    memory_df = memory_df[["HashApp", "AverageMemUsage"]]

    for cur_day in range(2, NUM_MEMORY_FILES + 1):
        print("generating memory data for day {}".format(cur_day))
        cur_df = pd.read_csv(data_path.replace("d01", "d{:02d}".format(cur_day)))
        cur_df = cur_df[["HashApp", "AverageAllocatedMb"]]
        cur_df.drop_duplicates(subset="HashApp", inplace=True)

        # Add another day's data to the columns, we match rows based on the function hash.
        memory_df = pd.merge(memory_df, cur_df, how="outer", on="HashApp", suffixes=(None, "_y"))

        memory_df["AverageMemUsage"] = memory_df.apply(lambda x : add_mem_val(x.AverageMemUsage,
                                                                             x.AverageAllocatedMb,
                                                                             cur_day), axis=1)

        del memory_df["AverageAllocatedMb"]

    memory_df.drop_duplicates(subset="HashApp", inplace=True)
    memory_df.to_pickle(save_path)

def add_mem_val(mem_val_list, mem_val, cur_day):
    if type(mem_val_list) != list:
        mem_val_list = [0] * (cur_day - 1)

    if pd.isna(mem_val):
        mem_val = 0
    
    mem_val_list.append(mem_val)

    return mem_val_list

if __name__ == '__main__':
    gen_memory()



