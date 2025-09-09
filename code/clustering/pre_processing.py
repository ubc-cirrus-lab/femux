import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append("..")
from clustering.utils import set_metric, set_75p_metric, gen_wasted_mem_baseline

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"
feature_file = data_dir + "features/features_{}/features_{}_{}_{}.pickle"
result_file = data_dir + "results/{}_{}_percent_{}/{}_cold_starts_wasted_mem.pickle"
exec_path = data_dir + "preproc_data/app_exec_time_data.pickle"

NUM_DAYS = 12
MINUTES_PER_DAY = 1440
NUM_FEATURE_FILES = 45
MS_PER_SEC = 1000

def get_train_test_data(forecasters, percentage, block_size, weight_mode, features):
    """Merge the block-level forecaster metrics and feature values into one dataframe

    forecasters: list[str]
    Names of forecasters to include
    
    mode: str
    "train" or "test" data

    returns:
        pd.Dataframe, pd.Dataframe
        train and test dataframes are returned in that order in the format required for clustering
    """
    num_blocks = NUM_DAYS * MINUTES_PER_DAY // block_size
    feature_df = combine_features(block_size, num_blocks, features) 
    dfs = []

    for mode in ["train", "test"]:
        df = combine_forecast_results(forecasters, percentage, mode, block_size, weight_mode)

        df = df.merge(feature_df, on="HashApp", how="left")

        dfs.append(format_data(df, num_blocks, features))

    return dfs[0], dfs[1]


def combine_features(block_size, num_blocks, features):
    """Combine all features into one dataframe
    """
    combined_df = pd.DataFrame()

    for feature in features:
        if feature == "ExecutionTime":
            continue

        feature_dfs = []

        for filenum in range(NUM_FEATURE_FILES):
            df = pd.read_pickle(feature_file.format(block_size, feature, block_size, filenum))

            # We want the number of harmonics in each block for each application
            if feature == "Harmonics":        
                df[feature] = df[feature].apply(lambda harmonics_per_block : [len(harmonics) for harmonics in harmonics_per_block])
            
            feature_dfs.append(df)

        feature_df = pd.concat(feature_dfs, ignore_index=True)        

        feature_df[feature] = feature_df[feature].apply(lambda x : x[:num_blocks])

        combined_df = feature_df if combined_df.empty else combined_df.merge(feature_df, on="HashApp")

    if "ExecutionTime" in features:
        exec_time_df = pd.read_pickle(exec_path)
        combined_df = combined_df.merge(exec_time_df, on="HashApp", how="left")
        # Get execution time for each block from a list of daily execution times
        combined_df["ExecutionTime"] = combined_df["ExecDurations"].apply(lambda x : [x[i * block_size // MINUTES_PER_DAY] / MS_PER_SEC for i in range(num_blocks)])
        combined_df.drop("ExecDurations", axis=1, inplace=True)
        print(combined_df)

    return combined_df

def combine_forecast_results(forecasters, percentage, mode, block_size, weight_mode):
    """Combine all forecaster results into one dataframe

    percentage: int
    percentage of data

    mode: str
    "train" or "test" data
    """
    forecast_df = pd.DataFrame()

    for forecaster in forecasters:
        df = pd.read_pickle(result_file.format(block_size, percentage, mode, forecaster))

        df = set_metric(df, weight_mode, block_size)

        df = df[["HashApp", "Metric"]]
        df.rename({"Metric": forecaster + "_Metrics"}, axis=1, inplace=True)
        
        forecast_df = df if forecast_df.empty else forecast_df.merge(df, on="HashApp")
    
    return forecast_df


def format_data(df, num_blocks, features):
        """ explode columns which have list of data in them
        e.g. row_0 = [1,2,3,4] => row_0_0 = 1, row_0_1 = 2, row_0_2 = 3, row_0_3 = 4

        Also drop all blocks where all forecasters only forecast 0

        df: dataframe with per-block features and forecaster data as lists, one row per application
            cols:
                forecaster metrics (e.g., AR_Metrics): list of metric values per block
                features (e.g., Harmonics): list of feature values per block
        
        returns: DataFrame
        df: has num_blocks*(len_original_rows) - # empty blocks
        """
        col_names = list(df.columns) + ["BlockIndex"]
        col_names.remove("HashApp")
        forecasters = [col for col in col_names if col.endswith("Metrics")]

        # Set a column for block indices of each app
        df["BlockIndex"] = df.apply(lambda x: [i for i in range(num_blocks)], axis=1)

        # Each app now has one row per block index with the associated feature and forecaster values
        df = df.explode(col_names)
        df.reset_index(drop=True, inplace=True)

        # Each block will know feature values of previous block, so we shift feature values
        # accordingly and drop the first block since it has no preceding block
        df = shift_rows(df, features)

        # Drop all blocks where all forecasters forecast only 0's to cut down clustering time
        df["Drop"] = df[forecasters].apply(lambda x : sum(x) == 0, axis=1)
        df = df[df.Drop == False]
        df.drop("Drop", axis=1, inplace=True)

        return df


def shift_rows(df, features):
    """ shift some rows by 1 to ensure a tested block is in sync with the next forecasted block
    df: DataFrame

    returns: DataFrame
    """
    new_dfs = []
    for curr_app in df['HashApp'].unique():
        partial_df = df[df['HashApp'] == curr_app]
        partial_df[features] = partial_df[features].shift(periods=1)
        partial_df = partial_df.iloc[1:]
        new_dfs.append(partial_df)
        
    new_df = pd.concat(new_dfs, axis = 0)

    return new_df