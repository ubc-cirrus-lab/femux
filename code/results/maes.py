import gc
import sys
import utils
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from time import strftime
from math import ceil

sys.path.append("../")

from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"


postproc_data_path = data_dir + "clustering/post_processed_data/processed_testing_df.pickle"
mae_save_path = data_dir + "results/mae/{}_maes.pickle"

os.makedirs(data_dir + "results/mae", exist_ok=True)

def mae_calc(file_subscript, forecaster, forecast_window, forecast_len, num_workers, data, 
                data_percentage, block_size, femux_path):
    """ calculate maes for femux and perfect femux

    forecaster: str
    "femux" for FeMux and "perfect_femux" to choose the best forecaster for each block

    contains following columns:
        TransformedValues: actual trace
        ForecastedValues: forecasted trace
        HashApp: Application hash

    returns: df
    modifies original df by adding a column with the number of cold starts at 
    each time step
    """
    df = utils.gather_data_for_results(data, data_percentage)
    df = utils.clean_data(df, block_size)
    
    print("adding forecaster values for {}".format(forecaster), strftime("%H:%M:%S"))
    df = utils.set_forecaster(df, forecaster, femux_path)
    
    if df.isnull().values.any():
        prev_len = len(df)
        df = df.dropna()
        num_missing = prev_len - len(df)

        raise Exception("Missing {} results for selected dataset".format(num_missing))

    print("generating results".format(forecaster), strftime("%H:%M:%S"))
    dfs = np.array_split(df, num_workers)
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(gen_maes_multiproc, dfs, [forecast_len]*num_workers, 
                                [forecast_window]*num_workers)

    result = pd.concat(results, ignore_index=True)

    print(result)
    result.to_pickle(mae_save_path.format(forecaster))
    del result
    del dfs
    
    gc.collect()


def gen_maes_multiproc(df, forecast_len, forecast_window):
    df["MAE"] = df.apply(lambda x : sum(gen_maes(x.TransformedValues,
                                                x.ForecastedValues,
                                                forecast_len,
                                                forecast_window)), axis=1)

    return df[["HashApp", "MAE"]]


def gen_maes(transformed_vals, forecasted_vals, forecast_len, num_past_elements):
    """Generate mean absolue error values for the forecasted value(s) at each timestep.
    transformed_vals: np.array(float)
        original trace (e.g., per-minute concurrency)
    
    forecasted_vals: np.array(np.array(float))
        each element is the forecasted value(s) at that given timestep

    forecast_len: int
        how many future elements to forecast

    num_past_elements:int
        number of past elements used for forecasting

    """
    trace_len = len(transformed_vals)
    num_forecasts = trace_len - forecast_len - num_past_elements
    mae_list = np.empty(shape=(num_forecasts, forecast_len))
    
    for trace_index in range(num_past_elements, trace_len - forecast_len):
        forecast_index = trace_index - num_past_elements
        mae_list[forecast_index] = get_mae(forecasted_vals[forecast_index], 
                                transformed_vals[trace_index: trace_index + forecast_len], 
                                forecast_len)
            
    return mae_list


if __name__ == '__main__':
    forecast_len = 1
    
    forecasters = ["AR_10", "FFT_10"]
    
    data_percentage = 100
    block_size = 504
    num_workers = 12
    femux_features = "Density_Linearity_Stationarity_Harmonics"
    femux_transformer = "StandardScaler"

    femux_path = ""
    
    for forecaster in forecasters:
        for data_mode in ["train"]:
            file_subscript = "{}_{}_percent_{}".format(block_size, data_percentage, data_mode)
            forecast_window = 120

            if forecaster == "IceBreaker" or forecaster == "FFT_10":
                forecast_window = 60
            elif "IdleTime" in forecaster or "HybridHist" in forecaster:
                forecast_window = 0

            mae_calc(file_subscript, forecaster, forecast_window, forecast_len, num_workers, data_mode, data_percentage, block_size, femux_path)
            gc.collect()