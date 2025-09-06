import pandas as pd
import numpy as np
import gc
import os
from concurrent.futures import ProcessPoolExecutor
from time import strftime
from concurrency_transformer import ConcurrencyTransformer
from container_idletime_transformer import ContainerIdleTimeTransformer
from utils import load_balance

preproc_data_path = "../../data/azure/preproc_data/invocation_data/preprocessed_data_00.pickle"
conc_event_path = "../../data/azure/transformed_data/conc_events/_app_conc_events_00.pickle"
conc_save_path = "../../data/azure/transformed_data/concurrency/_app_conc_00.pickle"

invocation_path = "../../data/azure/forecaster_data/invocations/{}_forecasts_00.pickle"
invocation_event_path = "../../data/azure/transformed_data/conc_events/{}_{}_conc_events_00.pickle"
invocation_conc_path = "../../data/azure/forecaster_data/concurrency/conc_{}_forecast_{}_00.pickle"

SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440            
NUM_SMALL_FILES = 40
NUM_MED_FILES = 4
NUM_LARGE_FILES = 1
NUM_PREPROC_FILES = 40

os.makedirs("../../data/azure/transformed_data/conc_events/", exist_ok=True)

def transform(num_workers, num_days, mode, size_filter, invocation_mode=False, forecaster="", func_mode=False):
    """Combines all invocations counts and execution durations into their respective lists.
    num_workers: int
    number of cores to use

    num_days: int
    number of days worth of data to process

    mode: str
    transformations are done at the function or app level and require events to have already been generated
        "idletime" to transform events into idletime series, requiresaverage concurrency as input
        "concurrency" to transform events into concurrency series, requires events as input
        "event" to generate concurrency events that are later used by idletime and concurrency transformers

    size_filter: str
    The number of invocations for the application
        "small": # invocations < 1M
        "medium": 1M < # invocations < 100M
        "large": < 100M

    icebreaker_mode: bool
    True if transforming IceBreaker forecasts, False otherwise.

    func_mode: bool
    True for function-level representation, False for application-level.

    Sideffect: All traces with 0 invocations are removed before processing
    """
    if invocation_mode == True and forecaster == "":
        raise Exception("Need forecaster name for transforming invocation forecasts")

    data_path, save_path, transformer, num_files = set_mode(mode, size_filter, invocation_mode, 
                                                             func_mode, forecaster)

    print("start time: ", strftime("%H:%M:%S"))
    num_minutes = [num_days * MINUTES_PER_DAY] * num_workers
    num_seconds = [num_days * MINUTES_PER_DAY * SECONDS_PER_MINUTE] * num_workers
    func_mode = [func_mode] * num_workers

    # once medium applications are generated, special handling for the transformation
    if size_filter == "medium" and mode == "concurrency":
        transform_medium(num_workers, num_seconds, data_path, save_path, transformer)
        return

    for chunk_index in range(num_files):
        df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(chunk_index)))
        
        df.dropna(axis=1, inplace=True)

        if df.empty:
            raise Exception("dataframe is empty after dropping all columns with nan values")
        
        if mode == "event":
            # some of the days in the Azure data include the same function twice
            if "HashFunction" in df.columns:
                df.drop_duplicates(subset=["HashFunction"], inplace=True)
            
            if size_filter == "large":
                df = preproc_large(data_path)
            else:
                df = filter_hashapps(df, size_filter)
            
            df.reset_index(drop=True, inplace=True)

            dfs = np.array_split(df, num_workers)

            fix_gaps(dfs, num_workers)

            with ProcessPoolExecutor(max_workers = num_workers) as executor:
                results = executor.map(transformer.transform_concurrency_events, dfs, num_minutes, func_mode)
    
        else:
            dfs = split_chunks(df, num_workers)

            with ProcessPoolExecutor(max_workers = num_workers) as executor:
                results = executor.map(transformer.transform, dfs, num_seconds)

        result = pd.concat(results, ignore_index=True)

        print("saving chunk {}".format(chunk_index))
        print(strftime("%H:%M:%S"))
        result.to_pickle(save_path.replace("_00", "_{:02d}".format(chunk_index)))

        # cleanup
        del result
        del results
        del dfs
        del df
        gc.collect()



def transform_medium(num_workers, num_seconds, data_path, save_path, transformer):
    save_indices = np.linspace(0, NUM_SMALL_FILES, NUM_MED_FILES + 1).astype(int)
    
    # split the big traces across different files
    for save_index in range(NUM_MED_FILES):
        event_df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(save_indices[save_index])))
        
        # combine smaller dataframes into a big one
        for data_index in range(save_indices[save_index] + 1, save_indices[save_index + 1]):
            next_event_df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(data_index)))
            event_df = pd.concat([event_df, next_event_df])
        
        dfs = np.array_split(event_df, num_workers)
        
        with ProcessPoolExecutor(max_workers = num_workers) as executor:
            results = executor.map(transformer.transform, dfs, num_seconds)
        
        result = pd.concat(results, ignore_index=True)
        
        print("saving big chunk {}".format(save_index))
        print(strftime("%H:%M:%S"))
        result.to_pickle(save_path.replace("_00", "_{:02d}".format(save_index)))
        
        del result
        del results
        del dfs
        del event_df
        gc.collect()
        

def preproc_large(data_path):
    """There aren't many large applications, so we combine them into one dataframe

    data_path: str
    size_filter: str

    returns: dataframe()
    dataframe containing all large applications
    """

    large_app_df = pd.read_pickle(data_path)

    large_app_df = filter_hashapps(large_app_df, "large")

    for chunk_index in range(1, NUM_SMALL_FILES):
        preproc_df = pd.read_pickle(data_path.replace("_00", "_{:02d}".format(chunk_index)))
    
        preproc_df = filter_hashapps(preproc_df, "large")

        large_app_df = pd.concat([large_app_df, preproc_df])
        
    return large_app_df



def split_chunks(df, num_workers):
    """Split the application-level dataframe into one sub-frame per worker, where
    sub-frames have roughly equal amounts of events (to even load between processors)

    df: dataframe
        contains columns "HashApp" and "NumEvents"
    
    num_workers: int
        number of sub-frames to split df into
    
    returns: list[dataframe]
        a list of sub-frames
    """
    if "HashFunction" not in df:
        print("Can't load balance, no HashFunction in Event dataframe")
        return np.array_split(df, num_workers)

    event_list = list(df["NumEvents"])
    event_sub_lists = [[] for _ in range(num_workers)]
    dfs = []

    # split hashapps across sublists so they have roughly even amounts of total events
    load_balance(event_list, event_sub_lists)

    # dictionary with numevents as key and rows as value
    events_dict = df.set_index("NumEvents").groupby("NumEvents").apply(lambda x : x.to_numpy().tolist()).to_dict()
    
    # split df into chunks according to the load balanced sub lists
    for sub_list in event_sub_lists:
        hashfunc_list = []
        for val in sub_list:
            hashfunc_list.append(events_dict[val][0][0])
            events_dict[val] = events_dict[val][1:]
        
        sub_df = df[df["HashFunction"].isin(hashfunc_list)].copy()

        dfs.append(sub_df)

    return dfs


def filter_hashapps(preproc_df, size_filter):
    if size_filter == "small":
        filter_df = preproc_df.loc[preproc_df["NumInvocations"] >= 1000000]
        filter_df = preproc_df[~preproc_df["HashApp"].isin(filter_df.HashApp)]
    
    elif size_filter == "medium":
        filter_df = preproc_df.loc[preproc_df["NumInvocations"] >= 100000000]
        filter_df = preproc_df[~preproc_df["HashApp"].isin(filter_df.HashApp)]

        filter2_df = filter_df.loc[(filter_df["NumInvocations"] >= 1000000)]
        filter_df = filter_df[filter_df["HashApp"].isin(filter2_df.HashApp)]
    
    elif size_filter == "large":
        filter_df = preproc_df.loc[preproc_df["NumInvocations"] > 100000000]
        filter_df = preproc_df[preproc_df["HashApp"].isin(filter_df.HashApp)]

    else:
        raise Exception("Need filter mode for transformation")

    return filter_df


def set_mode(mode, size_filter, invocation_mode, func_mode, forecaster):
    """ Set the transformation method, size of functions, and whether to transform at 
    function level or application level

    mode: str
        transformation type "event" transformation is a precursor to both idle time and 
        concurrency
    
    size_filter: str
    "small", "medium", "large", as described in transform()

    func_mode: bool
    True for function-level, False for app-level
    """
    num_files = set_num_files(size_filter, mode)
    
    if invocation_mode:
        event_path = invocation_event_path.format(size_filter, forecaster)
    else:
        event_path = conc_event_path.replace("_app", size_filter + "_app")

    if mode == "concurrency":
        if invocation_mode:        
            save_path = invocation_conc_path.format(size_filter, forecaster)    
        else:
            save_path = conc_save_path.replace("_app", size_filter + "_app")

        data_path = event_path
        transformer = ConcurrencyTransformer()

    elif mode == "event":
        data_path = invocation_path.format(forecaster) if invocation_mode else preproc_data_path
        save_path = event_path
        transformer = ConcurrencyTransformer()

    if func_mode:
        data_path = data_path.replace("app", "func")
        save_path = save_path.replace("app", "func")
    
    return data_path, save_path, transformer, num_files


def set_num_files(size_filter, mode):
    if mode == "event":
        return NUM_PREPROC_FILES if size_filter != "large" else 1

    num_files = NUM_SMALL_FILES

    if size_filter == "medium":
        num_files = NUM_MED_FILES
    elif size_filter == "large":
        num_files = NUM_LARGE_FILES
    
    return num_files


def fix_gaps(dfs, num_workers):
    """Put functions with the same HashApp in the same chunk.
    """
    for i in range(num_workers - 1):
        cur_df = dfs[i]
        
        if cur_df.empty:
                continue

        for j in range(i + 1, num_workers):
            next_df = dfs[j]

            # get the hashapp of the last function in the current chunk
            last_hashapp = cur_df["HashApp"].iloc[-1]

            # get the rows from the next chunk that match
            matching_rows = next_df.loc[next_df["HashApp"] == last_hashapp]
            
            if not matching_rows.empty:
                # add matching rows to current chunk
                dfs[i] = pd.concat([cur_df, matching_rows])

                # remove the matching rows from next chunk
                dfs[j] = next_df[next_df.HashApp != last_hashapp]

                cur_df = dfs[i]
            else:
                break


if __name__ == '__main__':   
    for size in ["small","medium", "large"]:
        transform(16, 14, "event", size)
    
    for size in ["small","medium", "large"]:
        transform(16, 14, "concurrency", size)