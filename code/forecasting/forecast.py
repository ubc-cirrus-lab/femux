import pandas as pd
import numpy as np
import warnings
import gc
from forecasting_sim import ForecastSimulation
from time import strftime

NUM_SMALL_CHUNKS = 40
NUM_MEDIUM_CHUNKS = 4
NUM_LARGE_CHUNKS = 1
CONC_FORECAST_WINDOW = 120
FFT_WINDOW = 60
IDLETIME_FORECAST_WINDOW = 10
TRAIN_SPLIT = 0.7

warnings.filterwarnings("ignore")

conc_forecast_path = "../../data/azure/forecaster_data/concurrency/conc_forecast_00.pickle"
conc_path = "../../data/azure/transformed_data/concurrency/app_conc_00.pickle"

def forecast(forecaster, forecast_len, forecast_param, data, num_workers, app_size, data_split, 
            proportion, weight_mode="default"):
    """Combines all invocations counts and execution durations into their respective lists.
    num_workers: int
    number of cores to use

    app_size: str
    which size of applications to forecast
        "small" for applications with functions that have <1M invocations
        "medium" for <100M invocations
        "large" for >100M invocations
    
    weight_mode: str
        "4_cs": 4 times cold start weight
        "4_wm": 4 times wasted memory weight
        "exec": include execution durations in RUM
    """
    
    data_path, save_path, num_files, sim = set_paths(data, app_size,
                                                    forecast_len, forecaster, forecast_param, weight_mode)

    print("{} start time: ".format(forecaster), strftime("%H:%M:%S"))

    print(save_path)

    filenums = get_num_files(data_split, proportion, num_files)

    for filenum in filenums:
        print("on chunk {}".format(filenum))
        transformed_df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(filenum)))

        result = sim.run_sim(transformed_df, num_workers)

        result.to_pickle(save_path.replace("_00", "_{:02d}".format(filenum)))

        # cleanup
        del result
        del transformed_df
        gc.collect()

        print("finished saving chunk {}".format(filenum))
        print(strftime("%H:%M:%S"))


def set_paths(data, app_size, forecast_len, 
              forecaster, forecast_param, weight_mode):
    print("Forecasting {} data".format(data))

    data_path = conc_path
    save_path = conc_forecast_path
    forecast_window = CONC_FORECAST_WINDOW

    data_path = data_path.replace("app_", "{}_app_".format(app_size))
    
    if weight_mode == "default":
        save_path = save_path.replace("_forecast", "_{}_forecast_{}".format(app_size, forecaster))
    else:
        save_path = save_path.replace("_forecast", "_{}_forecast_{}_{}".format(app_size, weight_mode, forecaster))

    if forecast_param != None:
        save_path = save_path.replace("_00", "_{}_00".format(forecast_param))

    if app_size == "small":
        num_files = NUM_SMALL_CHUNKS
    elif app_size == "medium":
        num_files = NUM_MEDIUM_CHUNKS
    elif app_size == "large":
        num_files = NUM_LARGE_CHUNKS

    forecast_window = FFT_WINDOW if forecaster == "IceBreaker" else forecast_window

    sim = ForecastSimulation(forecaster, weight_mode, forecast_param, forecast_len=forecast_len, num_past_elements=forecast_window, 
                                data_mode="concurrency")

    return data_path, save_path, num_files, sim


def get_num_files(data_split, proportion, num_files):
    """Sample data from train or test splits

    data_split: str
    "train" or "test", if "all" is chosen then generate all files

    proportion: int
    percentage of split to sample
    """

    num_train = int(np.ceil(TRAIN_SPLIT * num_files))
    num_test = num_files - num_train
    filenums = list(range(num_files))

    if data_split == "all":
        sample = filenums
    elif data_split == "train":
        sample = filenums[:int(np.ceil(num_train * proportion)) + 1]
    else:
        sample = filenums[-int(np.ceil(num_test * proportion)):]

    return sample


if __name__ == '__main__':
    forecast_len = 1
    data_mode = "azure"
    forecasters = [("FFT", 10), ("AR", 10), ("MarkovChain", None), ("10_min_keepalive", None), ("5_min_keepalive", None), ("IceBreaker", None)] #("Holt", None), ("ExpSmoothing", None), ("SETAR", 10)]
    num_workers = 16
    data_split = "train"
    proportion = 1

    weight_mode = "default"
    
    for forecaster in forecasters:
        for data_split in ["train", "test"]:
            for size in ["small", "medium", "large"]:
                forecast(forecaster[0], forecast_len, forecaster[1], data_mode, 
                num_workers, size, data_split, proportion, weight_mode)
