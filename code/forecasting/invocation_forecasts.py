import gc
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from time import strftime


import sys
sys.path.append("../forecasting/")
sys.path.append("../forecasting/forecasters/")
sys.path.append("../transform/")

from forecasting_sim import ForecastSimulation
from transform_azure import transform

preproc_data_path = "../../data/azure/preproc_data/invocation_data/preprocessed_data_{:02d}.pickle"
invocation_forecast_save_path = "../../data/azure/forecaster_data/invocations/{}_forecasts_{:02d}.pickle"
invocation_conc_path = "../../data/azure/forecaster_data/concurrency/conc_{}_forecast_{}_{:02d}.pickle"

NUM_CHUNKS = 40
SECONDS_PER_MINUTE = 60
FORECAST_LEN = 1
FORECAST_WINDOW = 60

num_files = {"small": 40, "medium": 4, "large": 1}
sizes = ["small", "medium", "large"]

os.makedirs("../../data/azure/forecaster_data/invocations", exist_ok=True)

def gen_forecasts(num_workers, forecaster, param):
    """ Use the IceBreaker forecasting implementation to get forecasted values for number of invocations
    per minute. This is per-function. 

    num_workers: int
        number of cores to use
    """
    print("Starting invocation forecasts for {}".format(forecaster))

    forecast_window = 60 if forecaster == "IceBreaker" else 120

    sim = ForecastSimulation(forecaster, param=param, forecast_len=FORECAST_LEN, num_past_elements=forecast_window)
    
    for chunk_index in range(NUM_CHUNKS):
        # format invocations per minute for forecasting simulator
        invocation_df = pd.read_pickle(preproc_data_path.format(chunk_index))
        invocation_df.rename({"InvocationsPerMin": "TransformedValues"}, axis=1, inplace=True)
        invocation_df["ContainerInvocationsPerMin"] = invocation_df.TransformedValues.tolist()

        forecast_df = sim.run_sim(invocation_df, num_workers)

        # preprocess data for transformation
        forecast_df = transformation_preprocess(forecast_df, forecast_window)
        forecast_df.to_pickle(invocation_forecast_save_path.format(format(forecaster, chunk_index)))
        
        print(forecast_df)
        # cleanup
        del forecast_df
        del invocation_df
        gc.collect()

        print("finished saving chunk {}".format(chunk_index))
        print(strftime("%H:%M:%S"))


def transformation_preprocess(forecast_df, forecast_window):
    """ adjust names and datastructures for transformation in next step
    """
    forecast_df.rename({"ForecastedValues": "InvocationsPerMin"}, axis=1, inplace=True)
    forecast_df["InvocationsPerMin"] = forecast_df.InvocationsPerMin.apply(lambda x: x.flatten())
    forecast_df["InvocationsPerMin"] = forecast_df.InvocationsPerMin.apply(lambda x: np.rint(x).astype(int))
    forecast_df["InvocationsPerMin"] = forecast_df.InvocationsPerMin.apply(lambda x: np.concatenate((np.zeros(forecast_window, dtype=int), 
                                        np.array(x))))
    forecast_df["InvocationsPerMin"] = forecast_df.InvocationsPerMin.apply(lambda x: np.append(x, 0))
    forecast_df["NumInvocations"] = forecast_df.InvocationsPerMin.apply(lambda x: sum(x))
    
    return forecast_df


def transform_forecasts(num_workers, forecaster):
    """ Convert IceBreaker's forecasts for number of invocations per minute, to average concurrency at an application level.
    """
    for size in sizes:
        transform(num_workers, 14, "event", size, invocation_mode=True, forecaster=forecaster)
        transform(num_workers, 14, "concurrency", size, invocation_mode=True, forecaster=forecaster)


def postproc_forecasts(forecaster, forecast_window):
    size = "small"

    for size in sizes:
        for filenum in range(num_files[size]):
            df = pd.read_pickle(invocation_conc_path.format(size, forecaster, filenum))
        
            df["ForecastedValues"] = df.TransformedValues.apply(lambda x : x[forecast_window:])
            df["ForecastedValues"] = df.ForecastedValues.apply(lambda x : np.array([[val] for val in x]))
            df["ContainerInvocationsPerMin"] = df.ContainerInvocationsPerMin.apply(lambda x : x[forecast_window:])

            df = df[["HashApp", "ForecastedValues", "ContainerInvocationsPerMin"]]

            df.to_pickle(invocation_conc_path.format(size, forecaster, filenum))


if __name__ == '__main__':
    forecaster = "IceBreaker"
    param = 10
    forecast_window = 60

    #gen_forecasts(28, forecaster, param)
    #transform_forecasts(5, forecaster)
    postproc_forecasts(forecaster, forecast_window)