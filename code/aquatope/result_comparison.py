from pathlib import Path
import pandas as pd
import numpy as np

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
result_dir = data_dir + "results/504_100_percent_test/"
femux_path = result_dir + "default_markov_v3_StandardScaler_femux_Density_Linearity_Stationarity_Harmonics_cold_starts_wasted_mem.pickle"
aquatope_path = result_dir + "Aquatope_cold_starts_wasted_mem.pickle"
static_ka_path = result_dir + "10_min_keepalive_cold_starts_wasted_mem.pickle"
preproc_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"

TOTAL_NUM_DAYS = 12
NUM_TRAINING_DAYS = 7
NUM_AQUATOPE_INPUT_STEPS = 48
BLOCK_SIZE = 504
STATIC_KEEPALIVE_WINDOW = 10
MINUTES_PER_DAY = 1440

START_INDEX = (BLOCK_SIZE * 21)
END_INDEX = (BLOCK_SIZE * 34)


def cs_pct_allocated_mem():
    aquatope_df = pd.read_pickle(aquatope_path)
    aquatope_df.dropna(inplace=True)
    femux_df = pd.read_pickle(femux_path)
    femux_df = femux_df[femux_df["HashApp"].isin(aquatope_df["HashApp"])]
    femux_df.dropna(inplace=True)
    static_ka_df = pd.read_pickle(static_ka_path)
    static_ka_df = static_ka_df[static_ka_df["HashApp"].isin(aquatope_df["HashApp"])]

    num_invocations = total_num_invocations(aquatope_df)
    _, static_ka_mem = get_cs_allocated_mem(static_ka_df, False)
    print("Total num invocations", num_invocations)
    print(femux_df)

    for i, df in enumerate([aquatope_df, femux_df]):
        cs, mem = get_cs_allocated_mem(df, i == 0) 

        print("\nForecaster is {}".format("Aquatope" if i == 0 else "Femux"))
        print("Cold start count", cs)
        print("Cold start %", 100 * cs / num_invocations)
        print("Mem allocated", mem)
        print("Mem allocated relative to 10-min KA", 100 * mem / static_ka_mem)
        

def get_cs_allocated_mem(df, is_aquatope=False):

    if not is_aquatope:
        df["NumColdStarts"] = df["NumColdStarts"].apply(lambda x : sum(x[20:]))
        df["MemAllocated"] = df["MemAllocated"].apply(lambda x : sum(x[20:]))

    return df.NumColdStarts.sum(), df.MemAllocated.sum()

    

def total_num_invocations(aquatope_df):    
    inv_df = pd.read_pickle(preproc_path.format(TOTAL_NUM_DAYS))
    inv_df = inv_df[inv_df["HashApp"].isin(aquatope_df["HashApp"])]
    inv_df["NumInvocations"] = inv_df["InvocationsPerMin"].apply(lambda x : x[START_INDEX: END_INDEX].sum())

    return inv_df["NumInvocations"].sum()



if __name__ == "__main__":
    cs_pct_allocated_mem()