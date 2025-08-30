import pickle
import sys
import pandas as pd

sys.path.append("../../")

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data") + "/"

cs_wm_forecaster_path = data_dir + "_cold_starts_wasted_mem.pickle"
hashapp_path = data_dir + "hashapps_by_size/{}_{}_apps.pickle"


def gen_result_df(forecasters, data_desc, gen_skip=True):
    result_df = pd.DataFrame()

    for forecaster in forecasters:
        print(forecaster)
        
        data_path = cs_wm_forecaster_path.replace("/_cold_starts", "/{}/{}_cold_starts".format(data_desc, forecaster))
        forecaster_df = pd.read_pickle(data_path)
        forecaster_df = forecaster_df[['HashApp', 'NumColdStarts', 'NumInvocations', 'MemAllocated', 'MemoryUsed']]

        #forecaster_df["SumColdStarts"] = forecaster_df.NumColdStarts.apply(lambda x: sum(x))
        #forecaster_df = forecaster_df[forecaster_df.SumColdStarts > 0]
        #forecaster_df.drop(["SumColdStarts"], inplace=True, axis=1)

        result_df = update_result_df(result_df, forecaster_df, forecaster)
    if gen_skip:    
        result_df["SkipBlocks"] = result_df.MemAllocated.apply(lambda x : gen_skip_blocks(forecasters, x))

    return result_df


def gen_skip_blocks(forecasters, mem_allocated):
    """Determine which blocks should be skipped due to no traffic (both actual
    and forecaster)

    forecasters: list[str]

    mem_allocated: list[float]
    Amount of memory allocated per block in GBs

    returns: list[bool]
    True if block should be skipped
    """
    
    skip_blocks = []

    for block_index in range(33):
        sum_mem_used = 0

        for forecaster in forecasters:
            sum_mem_used += mem_allocated[forecaster][block_index]

        skip_block = True if sum_mem_used < 0.0000001 else False
        skip_blocks.append(skip_block)

    return skip_blocks


def update_result_df(result_df, forecast_df, forecaster):
    """For each application, aggregate all forecaster data into a dict for each app"""

    if result_df.empty:
        result_df = forecast_df
        
        for col in result_df.columns:
            if col != "HashApp":
                result_df[col] = result_df[col].apply(lambda x : {forecaster: x})
        
        return result_df

    result_df = result_df.merge(forecast_df, on="HashApp", how="left", suffixes=(None, "_y"))
    dropped_df = result_df.dropna()

    if len(dropped_df) < len(result_df):
        print("Dropped {} apps due to missing values in {}".format(len(result_df) - len(dropped_df), forecaster))
        result_df = dropped_df

    
    for col in result_df.columns:
        if col == "HashApp" or col.endswith("_y"):
            continue
        
        result_df[col] = result_df.apply(lambda x : update_dict(x[col], x[col + "_y"], forecaster), axis=1) 

        del result_df[col + "_y"]

    return result_df


def update_dict(result_dict, new_vals, forecaster):
    result_dict[forecaster] = new_vals
    return result_dict


def parse_size(df, mode, size):
    """mode: str
    "test" or "train
    """
    
    with open(hashapp_path.format(size, mode), "rb") as f:
        size_list = pickle.load(f)

    return df[df.HashApp.isin(size_list)] 