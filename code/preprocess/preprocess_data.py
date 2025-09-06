import pandas as pd
import numpy as np
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = "../../data/azure/azure_data/"
SAVE_DIR = "../../data/azure/preproc_data/invocation_data/"

invocation_filename = "invocations_per_function_md.anon.d01.csv"
execution_filename = "function_durations_percentiles.anon.d01.csv"
stitched_filename = "stitched_functions_.pickle"
combined_filename = "combined_function_invocations_and_durations_d01.pickle"
preprocessed_filename = "preprocessed_data_.pickle"

combined_path = SAVE_DIR + stitched_filename
preproc_path = SAVE_DIR + preprocessed_filename
corrupt_hashapp_path = "../../data/azure/preproc_data/corrupt_hashapps.pickle"

MINUTES_PER_DAY = 1440
NUM_CHUNKS = 40

os.makedirs(SAVE_DIR, exist_ok=True)

"""Generate csv file(s) with each function's data across 12 days stored in sequential columns."""
def gen(num_days, num_workers=4):
    invocation_filepaths = gen_filepaths(invocation_filename, DATA_DIR, num_days)
    execution_filepaths = gen_filepaths(execution_filename, DATA_DIR, num_days)
    combined_filepaths = gen_filepaths(combined_filename, SAVE_DIR, num_days)
    
    # Add invocation duration to invocation counts of each function (these are in separate files)
    add_execution_times(invocation_filepaths, execution_filepaths, combined_filepaths)

    # Glue together each function's data across all 12 days.
    stitch(combined_filepaths)

    combine_data(num_workers, num_days)

    fix_gaps()


"""Generate filepaths for all 12 days of Azure Data
returns -- list[str] filepaths of Azure Data
"""
def gen_filepaths(filename, dir, num_days):
    filepaths = []
    for i in range(1, num_days + 1):
        filepath = dir + filename.replace("d01", "d{:02d}".format(i))
        filepaths.append(filepath)

    return filepaths


"""Add function duration data to function invocation count data. 
Will overwrite if files already exist.
"""
def add_execution_times(invocation_filepaths, execution_filepaths, combined_filepaths):
    corrupt_hashapps = set()
    average_null = set()
    average_inv_null = set()
    avg_zero = set()

    for i in range(len(execution_filepaths)):
        execution_file = execution_filepaths[i]
        invocation_file = invocation_filepaths[i]
        combined_file = combined_filepaths[i]
        
        invocation_df = pd.read_csv(invocation_file)
        invocation_df.drop(columns=["HashOwner", "Trigger"], inplace=True)
        exec_df = pd.read_csv(execution_file)

        # Only need the Average Execution Time data, and keep function hash for merging data.
        exec_df = exec_df[["HashFunction", "Average"]]

        combined_df = pd.merge(invocation_df, exec_df, how="left", on="HashFunction")

        # Mark all apps with missing data or recorded execution times of 0 as corrupt 
        corrupt_hashapps.update(combined_df[combined_df.isnull().any(axis=1)].HashApp.tolist())
        average_inv_null.update(combined_df[combined_df.isnull().any(axis=1)].HashApp.tolist())
        average_null.update(combined_df[combined_df.Average.isnull()].HashApp.tolist())
        corrupt_hashapps.update(combined_df[combined_df.Average == 0].HashApp.tolist())
        avg_zero.update(combined_df[combined_df.Average == 0].HashApp.tolist())
        
        combined_df.to_pickle(combined_file)

        print("Combined day {} of data".format(i+1))

    with open(corrupt_hashapp_path, 'wb') as f:
        pickle.dump(list(corrupt_hashapps), f)


"""Store function data across all 12 days in sequential order.
We iteratively merge existing data with the next day based on function hashes (HashFunction).
list - data_filepaths : list of paths to pickle files that store each day of data.
"""
def stitch(data_filepaths):
    # Our first gen_file is just day 1 data
    stitch_df = pd.read_pickle(data_filepaths[0])
    

    # For each function, add the data for each of the 12 days to its columns.
    for cur_day, data_file in enumerate(data_filepaths[1:]):
        data_df = pd.read_pickle(data_file)

        # Add another day's data to the columns, we match rows based on the function hash.
        stitch_df = pd.merge(stitch_df, data_df, how="outer", on="HashFunction", suffixes=(None, "_{}".format(2)))

        if "HashApp_2" in stitch_df:
            stitch_df["HashApp"] = stitch_df.apply(lambda x : add_hashapp(x.HashApp, x.HashApp_2), axis=1)
        
            stitch_df.drop(columns=["HashApp_2"], inplace=True)

        print("Finished stitching day {}".format(cur_day + 2))

    stitch_df.sort_values(by = ["HashApp"], inplace=True, ignore_index = True)

    stitch_df.replace(np.nan, 0, inplace=True)

    stitch_dfs = np.array_split(stitch_df, NUM_CHUNKS)

    # save chunks separately
    for i, stitch_df in enumerate(stitch_dfs):
        stitch_df.to_pickle(combined_path.replace("_.", "_{:02d}.".format(i)))


def add_hashapp(hashapp, new_hashapp):
    if pd.isna(hashapp):
        return new_hashapp
    return hashapp


def combine_data(num_workers, num_days):
    """Combines all invocations counts and execution durations into their respective lists.
    """
    num_days = [num_days] * num_workers
    
    # we reformat each chunk sequentially, but the reformatting is done in parallel
    for i in range(NUM_CHUNKS):
        stitch_df = pd.read_pickle(combined_path.replace("_.", "_{:02d}.".format(i)))
        
        print("reformatting chunk {}".format(i))
        dfs = np.array_split(stitch_df, num_workers)

        with ProcessPoolExecutor(max_workers = num_workers) as executor:
            results = executor.map(reformat_data, dfs, num_days)

        result = pd.concat(results, ignore_index=True)

        result.to_pickle(preproc_path.replace("_.", "_{:02d}.".format(i)))



def reformat_data(stitch_df, num_days):
    """Combine all invocation counts, and store a list for the number
    of executions and average execution duration per day.
    """
    stitch_df.reset_index(drop=True, inplace=True)
    num_traces = len(stitch_df)
    invocation_counts = [np.empty(0, dtype=np.int32) for _ in range(num_traces)]
    num_invocations = [0] * num_traces
    exec_durations = [np.empty(0, dtype=np.int32) for _ in range(num_traces)]
    hashapp_list = [""] * num_traces

    for index, row in stitch_df.iterrows():
        for i in range(2, num_days * MINUTES_PER_DAY + num_days, MINUTES_PER_DAY + 1):
            cur_invocation_counts = list(row[i:i + MINUTES_PER_DAY])
            invocation_counts[index] = np.append(invocation_counts[index], [int(num) for num in cur_invocation_counts])
            exec_durations[index] = np.append(exec_durations[index], int(row[i + MINUTES_PER_DAY]))
            
        num_invocations[index] = int(sum(invocation_counts[index]))

    stitch_df["InvocationsPerMin"] = invocation_counts
    stitch_df["NumInvocations"] = num_invocations
    stitch_df["ExecDurations"] = exec_durations

    stitch_df = stitch_df[["HashFunction", "HashApp", "InvocationsPerMin", "NumInvocations", "ExecDurations"]]

    return stitch_df


def fix_gaps():
    for chunk_index in range(NUM_CHUNKS - 1):
        print("stitching hashapp for chunk {}".format(chunk_index))
        cur_chunk_path = preproc_path.replace("_.", "_{:02d}.".format(chunk_index))
        next_chunk_path = preproc_path.replace("_.", "_{:02d}.".format(chunk_index + 1))
        
        cur_preproc_df = pd.read_pickle(cur_chunk_path)
        next_preproc_df = pd.read_pickle(next_chunk_path)

        # get the hashapp of the last function in the current chunk
        last_hashapp = cur_preproc_df["HashApp"].iloc[-1]

        # get the rows from the next chunk that match
        matching_rows = next_preproc_df.loc[next_preproc_df["HashApp"] == last_hashapp]
        
        if not matching_rows.empty:
            # add matching rows to current chunk
            cur_preproc_df = pd.concat([cur_preproc_df, matching_rows], ignore_index=True)

            # remove the matching rows from next chunk
            next_preproc_df = next_preproc_df[next_preproc_df.HashApp != last_hashapp]

        cur_preproc_df.to_pickle(cur_chunk_path)
        next_preproc_df.to_pickle(next_chunk_path) 


if __name__ == '__main__':
    gen(14, num_workers=16)
