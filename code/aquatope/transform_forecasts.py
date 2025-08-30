import gc, os, sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time


sys.path.append("..")
from transform.concurrency_transformer import ConcurrencyTransformer

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
inv_forecast_path = data_dir + "forecaster_data/invocations/aquatope_forecasts_{}.pickle"
conc_forecast_path = data_dir + "forecaster_data/concurrency/conc_forecast_{}_{:02d}.pickle"
exec_path = data_dir + "preproc_data/app_exec_time_data.pickle"


NUM_TRAINING_DAYS = 7
NUM_DAYS = 5
MINUTES_PER_DAY = 1440
SECONDS_PER_MIN = 60
INPUT_STEPS = 48
OUTPUT_STEPS = 1


def transform(split, num_workers):
    """ Transform invocation per minute data to average concurrency per minute data.
    """
    if os.path.exists(conc_forecast_path.format("aquatope", split)):
        print("skipping split {}".format(split))
        return

    df = prep_forecasts(split)
    
    dfs = np.array_split(df, num_workers)
    num_minutes = len(df.InvocationsPerMin.tolist()[0])

    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(full_transform, dfs, [num_minutes] * num_workers)

    results = pd.concat(results)
    print(df)

    results[["HashApp", "TransformedValues"]].to_pickle(conc_forecast_path.format("aquatope", split))


def prep_forecasts(split):
    df = pd.read_pickle(inv_forecast_path.format(split))
    df = pd.merge(df, pd.read_pickle(exec_path), on="HashApp")
    
    df["InvocationsPerMin"] = df.ForecastedValues.apply(lambda x : [np.ceil(max(0, y)).astype(int) for y in x])
    df["NumInvocations"] = df.InvocationsPerMin.apply(lambda x : sum(x))
    df["ExecDurations"] = df.ExecDurations.apply(lambda x : x[NUM_TRAINING_DAYS:])
    del df["ForecastedValues"]
    
    return df


def full_transform(df, num_minutes):
    transformer = ConcurrencyTransformer()
    prev = time()

    df = transformer.transform_concurrency_events(df, num_minutes, func_mode=True)
    df = transformer.transform(df, num_minutes * SECONDS_PER_MIN)

    return df


if __name__ == "__main__":
    for split in range(100):
        transform(split, 14)
