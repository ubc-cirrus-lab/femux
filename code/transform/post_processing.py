import pandas as pd
import gc
from time import strftime
from concurrency_transformer import ConcurrencyTransformer

conc_event_path = "../data/transformed_data/app_conc_events/_app_conc_events_00.pckl"
conc_save_path = "../data/transformed_data/concurrency/_app_conc_00.pckl"
idletime_save_path = "../data/transformed_data/idletime/_app_idle_00.pckl"

SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440            
NUM_SMALL_CHUNKS = 40
NUM_MED_CHUNKS = 4
NUM_LARGE_CHUNKS = 1


def process_multiminute(num_days, mode, filter_mode, timestep):
    """Combines all invocations counts and execution durations into their respective lists.
    num_days: int
    number of days worth of data to process

    mode: str
    how the new timsteps are calculated
        "max": maximum value from the timesteps being merged
        "avg": average value from timesteps being merged

    filter_mode: str
    size of the applications being processed
        "small": applications with functions that have <1M invocations
        "medium": 1M < invocations < 100M
        "large": > 100M

    """

    data_path, save_path, transformer, num_chunks = set_mode(mode, filter_mode, timestep)

    print("start time: ", strftime("%H:%M:%S"))
    num_minutes = num_days * MINUTES_PER_DAY



    for chunk_index in range(num_chunks):
        df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(chunk_index)))

        df["TransformedValues"] = df.TransformedValues.apply(lambda x : transformer.multi_minute_concurrency(x, 
                                                                num_minutes, timestep, mode))
        
        df.to_pickle(save_path.replace("_00", "_{:02d}".format(chunk_index)))

        del df
        gc.collect()

        print("finished saving chunk {}".format(chunk_index))
        print(strftime("%H:%M:%S"))


def set_mode(mode, filter_mode, new_timestep):
    data_path = conc_save_path.replace("_app", filter_mode + "_app")
    save_path = data_path.replace("concurrency/", "concurrency/{}_minute/{}/".format(new_timestep, mode))
    num_chunks = set_num_chunks(filter_mode)

    return data_path, save_path, ConcurrencyTransformer(), num_chunks


def set_num_chunks(filter_mode):
    num_chunks = NUM_SMALL_CHUNKS

    if filter_mode == "medium":
        num_chunks = NUM_MED_CHUNKS
    elif filter_mode == "large":
        num_chunks = NUM_LARGE_CHUNKS
    
    return num_chunks


if __name__ == '__main__':
    for size in ["small", "medium", "large"]:
        for timestep in [5, 10, 20]:
            process_multiminute(14, "max", size, timestep)
