import re
import os
import pandas as pd
import numpy as np
import time
from os import listdir
from concurrent.futures import ProcessPoolExecutor
from os.path import isfile, join


num_file_dict = {"small":40 , "medium": 4, "large": 1}
IDLETIME_FORECAST_WINDOW = 10

idletime_forecast_path = "../../data/azure/forecaster_data/idletime/idletime_{}_forecast_{}_{:02d}.pickle"
conc_path = "../../data/azure/transformed_data/concurrency/{}_app_conc_{:02d}.pickle"
conc_forecast_path = "../../data/azure/forecaster_data/concurrency/conc_{}_forecast_{}_{:02d}.pickle"


def combine_tv_and_fv():
    INPUT_FOLDER1 = "../../data/forecaster_data/fft_concurrency_only_corrected_concurrency/"
    INPUT_FOLDER2 = "../../data/concurrency_small/"

    forecast_file_names = [
        f for f in listdir(INPUT_FOLDER1) if isfile(join(INPUT_FOLDER1, f))
    ]
    forecast_file_names.sort()

    conc_file_names = [
        f for f in listdir(INPUT_FOLDER2) if isfile(join(INPUT_FOLDER2, f))
    ]
    conc_file_names.sort()
    ind = 0
    for forecast_file_name, conc_file_name in zip(forecast_file_names, conc_file_names):
        start = time.time()
        fcast_df = pd.read_pickle(INPUT_FOLDER1 + forecast_file_name)
        conc_df = pd.read_pickle(INPUT_FOLDER2 + conc_file_name)
        conc_df = pd.merge(conc_df, fcast_df, on='HashApp', how="left")
        conc_df.to_pickle(INPUT_FOLDER1 + forecast_file_name)
        print("completed file number {} in {}".format(ind, time.time()-start))
        ind += 1

def get_fv():
    INPUT_FOLDER = "../../data/forecaster_data/fft_concurrency/"
    OUTPUT_FOLDER = "../../data/forecaster_data/fft_concurrency_only_corrected_concurrency/"

    forecasted_file_names = [
        f for f in listdir(INPUT_FOLDER) if isfile(join(INPUT_FOLDER, f))
    ]

    forecasted_file_names.sort()
    # conc_file_names.sort()
    print(forecasted_file_names)
    # print(conc_file_names)
    # forecasted_file_names = sorted(forecasted_file_names, key=lambda x: int(re.findall(r"\d+", x)[0]))
    ind = 0
    for file_name in forecasted_file_names:
        start = time.time()
        pre_df = pd.read_pickle(INPUT_FOLDER + file_name)
        app_combined_forecast = []
        for curr_app in pre_df['HashApp'].unique():
            partial_df = pre_df[pre_df['HashApp'] == curr_app]
            local_app_combined_forecast = [[0]]*20039
            for i in range(len(local_app_combined_forecast)):
                for _, row in partial_df.iterrows():
                    partial_app_level_fcast = row['ForecastedValues']
                    local_app_combined_forecast[i][0] = max(local_app_combined_forecast[i][0], partial_app_level_fcast[i][0])
            app_combined_forecast.append(local_app_combined_forecast)
        new_df = pd.DataFrame()
        all_apps = [app_name for app_name in pre_df['HashApp'].unique()]
        new_df['HashApp'] = all_apps
        new_df['ForecastedValues'] = app_combined_forecast
        print(new_df)
        new_df.to_pickle(OUTPUT_FOLDER + "conc_{}_forecast_{}_{:02}.pickle".format("small", "FFT", ind))
        print("file {} finished in {}".format(ind, time.time()-start))
        ind += 1

def get_combined_transformed_and_forecasted_values(pre_df, forecast_len):
    # combine transformed values
    app_level_transformed_values = []
    for curr_app in pre_df['HashApp'].unique():
        curr_df = pre_df[pre_df['HashApp'] == curr_app]
        per_app_transformed_values = [0]*len(curr_df.at[curr_df.index.tolist()[0], 'TransformedValues'])
        for _, row in curr_df.iterrows():
            for i in range(len(row.TransformedValues)):
                per_app_transformed_values[i] = max(per_app_transformed_values[i], row.TransformedValues[i])
        app_level_transformed_values.append(per_app_transformed_values)
    # combine forecasted values
    app_combined_forecast = []
    for curr_app in pre_df['HashApp'].unique():
        partial_df = pre_df[pre_df['HashApp'] == curr_app]
        local_app_combined_forecast = [[0]*forecast_len for _ in range(len(partial_df.at[partial_df.index.tolist()[0], 'ForecastedValues']))]
        for i in range(len(local_app_combined_forecast)):
            for j in range(forecast_len):
                for _, row in partial_df.iterrows():
                    partial_app_level_fcast = row['ForecastedValues']
                    local_app_combined_forecast[i][j] = max(local_app_combined_forecast[i][j], partial_app_level_fcast[i][j])
        app_combined_forecast.append(local_app_combined_forecast)
    new_df = pd.DataFrame()
    all_apps = [app_name for app_name in pre_df['HashApp'].unique()]
    new_df['HashApp'] = all_apps
    new_df['TransformedValues'] = app_level_transformed_values
    new_df['ForecastedValues'] = app_combined_forecast
    print(new_df)
    return new_df


def gen_multiproc_conc_forecasts(df):
    df["ForecastedValues"] = df.apply(lambda x : gen_conc_forecasts(x.ForecastedValues, 
                                                                x.TransformedValues, x.HashApp), axis=1)

    return df                                                        


def gen_conc_forecasts(it_forecasts, avg_conc_per_min, hashapp):
    num_minutes = len(avg_conc_per_min)
    forecasts = np.zeros((num_minutes, 1))

    if not isinstance(it_forecasts, np.ndarray):
        return np.zeros((num_minutes, 1))

    num_active = 0
    cur_idletime_pred = 0
    next_active_minute = -1
    
    for cur_minute in range(num_minutes):
        forecasts[cur_minute] = [1] if cur_minute == next_active_minute else [0]

        if avg_conc_per_min[cur_minute] == 0:
            continue

        num_active += 1

        # we need 2 invocations to calculate the first idletime 
        if num_active > 1:
            it_forecast_index = min(num_active - 2, len(it_forecasts) - 1)

            # the idletime forecast predicts number of idle minutes, so we add 1 to get
            # the next predicted minute that has traffic. next_active_minute is updated if it 
            # was predicted to come after this active minute
            next_active_minute = it_forecasts[it_forecast_index] + cur_minute + 1

    return np.array(forecasts)
