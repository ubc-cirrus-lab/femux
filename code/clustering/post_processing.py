import pandas as pd
import numpy as np
import gc
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from time import strftime
sys.path.append("../")
from results.utils import gather_data_for_results
from clustering.utils import add_forecaster_values, add_idletime_forecasts
from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"

clustering_data_path = data_dir + "clustering/testing_output/testing_{}_{}_{}_{}_{}_percent_df_{}.pickle"
postproc_save_path = data_dir + "clustering/post_processed_data/processed_testing_df_{}_{}_{}_{}_{}_percent_{}.pickle"

os.makedirs(data_dir + "clustering/post_processed_data/", exist_ok=True)

NUM_DAYS = 12
MINUTES_PER_DAY = 1440

ALL_FEATURES = ["Stationarity", "Density", "Linearity", "Harmonics"]

def post_process(perfect_femux, classification_model, default_forecaster, block_size, forecast_window,
                forecast_len, num_workers, features, percentage, transformer, weight_mode):
    """ generates the values FeMux would forecast based on the clustering results
    
    perfect_femux: bool
    True if generating perfect FeMux

    default_forecaster: str
    name of forecaster to use during the first block (which FeMux needs to characterize)
    
    block_size: int
    
    forecast_window: int
    number of observations considered

    forecast_len: int

    returns: dataframe
    saves dataframe for save_path with the following columns:
        HashApp: Application's HashApp
        ForecastedValues: Values that FeMux would forecast at each element
    """
    print("Starting post process for femux with {} features and {} transformer for weight mode {}".format(features, transformer, weight_mode))
    data_path, save_path, col_name = set_postproc_mode(perfect_femux, classification_model, features, weight_mode, transformer, block_size, percentage)
    
    df = pd.read_pickle(data_path)

    if df.isnull().values.any(): 
        raise Exception("Dataframe contains nan {}".format(df[df.isnull().any(axis=1)]))
    
    df, forecasters = get_forecaster_names(df, default_forecaster, col_name)

    print("adding forecaster values for: {}".format(forecasters), strftime("%H:%M:%S"))
    for forecaster in forecasters:
        df = add_forecaster_values(df, forecaster)
        gc.collect()

    df = forecaster_vals_to_dict(df, forecasters)      

    dfs = np.array_split(df, num_workers)
    
    print("generating forecasted values", strftime("%H:%M:%S"))
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(multiproc_gen, dfs, [default_forecaster] * num_workers, [col_name] * num_workers, 
                                [block_size] * num_workers, [forecast_window] * num_workers, 
                                [forecast_len] * num_workers)
    
    result = pd.concat(results, ignore_index=True)

    result["ForecastedValues"] = result.ForecastedValues.apply(lambda x : np.array(x))

    print("Saving postproc to: {}".format(save_path))
    result.to_pickle(save_path)

    print(result)

    del result
    del results
    gc.collect()


def multiproc_gen(df, default_forecaster, col_name, block_size, forecast_window, forecast_len):
    df["ForecastedValues"] = df.apply(lambda x: gen_femux_forecasted_vals(default_forecaster,
                                                                            x.BlockIndex,
                                                                            x[col_name],
                                                                            x.ForecastedValues,
                                                                            block_size,
                                                                            forecast_window,
                                                                            forecast_len), axis=1)

    df = df[["HashApp", "ForecastedValues"]]

    return df


def gen_femux_forecasted_vals(default_forecaster, block_indices, chosen_forecasters, forecasted_vals_dict, 
                            block_size, forecast_window, forecast_len):
    """ Produces the values that FeMux would forecast at each time step by 
    filling each block with the forecasted values from the forecaster that FeMux has chosen
    for that block. The first block is filled with forecasts from the default forecaster
    since FeMux needs to characterize the prior block to select a forecaster.

    block_indices: list[int]
    list of block indices that were included in clustering: we discard blocks where all forecasters only 
    forecast 0's

    chosen_forecasters: list[str]
    the chosen forecaster for each block (in consecutive order of blocks)

    forecasted_vals_dict: dict{str: list[list[float]]}
    key is the forecaster name, value is the list of forecasted values. Each forecasted
    value is a list to allow for multistep forecasting

    block_size, forecast_window, forecast_len explained above
    
    returns:
    list[list[float]]
    values forecasted by FeMux at each timestep 

    """
    num_elements = NUM_DAYS * MINUTES_PER_DAY - forecast_window - forecast_len

    # forecaster data already excludes the forecast_window, so our offset into 
    # the second block has to account for that
    forecaster_offset = block_size - forecast_window

    # fill the first block of forecasted values with the default forecaster
    # set all forecasts to 0 to start
    femux_forecasts = [[0] for _ in range(num_elements)]
    femux_forecasts[:forecaster_offset] = forecasted_vals_dict[default_forecaster][:forecaster_offset]
    
    # add the forecasted values selected by FeMux for the second block onwards
    # we update forecasts for all blocks that have actual forecasts, which are 
    # stored in block_indices. 
    for forecaster_index, cur_block in enumerate(block_indices):
        cur_index = cur_block * block_size - forecast_window
        stop_index = min(cur_index + block_size, num_elements)

        cur_forecaster = chosen_forecasters[forecaster_index].replace("_Metrics", "")
        
        # FFT forecasting window is 60 instead of 120
        if cur_forecaster == "FFT_10":
            cur_index += 60
        
        femux_forecasts[cur_index:stop_index] = forecasted_vals_dict[cur_forecaster][cur_index:stop_index]

    return femux_forecasts


def forecaster_vals_to_dict(df, forecasters):
    # initialize new column with empty dicts
    df["ForecastedValues"] = [{} for _ in range(len(df))]

    for forecaster in forecasters:
        forecaster_col_name = "ForecastedValues_" + forecaster

        df["ForecastedValues"] = df.apply(lambda x : add_to_dict(x.ForecastedValues,
                                                            x[forecaster_col_name],
                                                            forecaster), axis=1)
        
        del df[forecaster_col_name]
    return df    


def add_to_dict(forecast_dict, forecasted_vals, forecaster):
    forecast_dict[forecaster] = forecasted_vals

    return forecast_dict


def set_postproc_mode(perfect_femux, classification_model, features, weight_mode, transformer, block_size, percentage):
    Path(data_dir + "clustering/post_processed_data/").mkdir(parents=True, exist_ok=True)
    feature_names = "_".join(features)
    
    save_path = postproc_save_path.format(classification_model, weight_mode, transformer, block_size, percentage, feature_names)
    data_path = clustering_data_path.format(classification_model, weight_mode, transformer, block_size, percentage, feature_names)

    if perfect_femux:
        save_path = save_path.replace("processed_testing", "processed_perfect_femux_testing")
        col_name = "Oracle_Forecaster"
    else:
        col_name = "Cluster_Forecaster"
    
    return data_path, save_path, col_name


def get_forecaster_names(df, default_forecaster, col_name):
    """ Get the names of forecasters that will be used in FeMux.

    df: pd.dataframe()
        Dataframe including the chosen forecaster for each block
    
    default_forecaster: str
        Forecaster to be used before first block is complete
    
    col_name: str
        "Oracle_Forecaster": Choose best forecaster for each block
        "Cluster_Forecaster": Choose forecaster based on ML model

    returns: 
        df: dataframe with traces and list of forecasters (for each block)
        forecasters: forecasters that are chosen
    """

    # get names of forecasters that FeMux uses
    forecasters = df[col_name].unique() 
    forecasters = [forecaster.replace("_Metrics", "") for forecaster in forecasters]
    
    # default forecaster used for first block of data
    if default_forecaster not in forecasters:
        forecasters.append(default_forecaster)

    df = df.groupby("HashApp").agg({col_name: list, "BlockIndex": list})
    df.sort_values(by="HashApp", inplace=True)
    df.reset_index(inplace=True)
    
    return df, forecasters


if __name__ == '__main__':
    post_process(False, "kmeans", "10_min_keepalive", block_size=504, forecast_window=120, forecast_len=1, 
            num_workers=1, features=["Density", "Linearity", "Stationarity", "Harmonics"], percentage=100, 
            transformer="StandardScaler", weight_mode="default")