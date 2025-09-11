import pandas as pd
import numpy as np
import concurrent.futures
import math
import os
import HypothesisTests as ht
import warnings
from os import listdir
from os.path import isfile, join
from pathlib import Path

warnings.filterwarnings("ignore")

data_dir = str(Path(__file__).resolve().parents[2] / "data" / "azure") + "/"

exec_path = data_dir + "preproc_data/app_exec_time_data.pickle"

class HypothesisTesting:
    def __init__(self, max_workers, block_size) -> None:
        self.max_workers = max_workers
        self.block_size = block_size
        self.MINUTES_PER_DAY = 1440
        self.number_of_days = 14
        # this flag is set internally do not touch
        self.func_flag = False
        
        self.OUTPUT = "../../data/azure/features/features_{}/".format(self.block_size)

        if not os.path.exists(self.OUTPUT):
            print("creating directory structure...")
            os.makedirs(self.OUTPUT)

    def process_transformed_traces(self, transformed_path, transformed_func_path, feature_types):
        for feature_type in feature_types:
            if feature_type == "ExecutionTimes":
                exec_time_df = pd.read_pickle(exec_path)
                print(exec_time_df)
            else:   
                transformed_file_names = [
                    f for f in listdir(transformed_path) if isfile(join(transformed_path, f))
                ]
                transformed_file_names.sort()
                print("files under consideration per app: ", transformed_file_names)
                
                for index, file_name in enumerate(transformed_file_names):
                    print("starting with file: {}".format(file_name))
                    transformed_df = pd.read_pickle(transformed_path + file_name)
                    result = self.multiprocessor_apply_tests(transformed_df, feature_type)
                    result.to_pickle(self.OUTPUT+"features_{}_{}_{}.pickle".format(feature_type, self.block_size, index))
                    print("completed file: {}".format(file_name))
                

    """
    Perform Hypothesis Testing on the DF and return the results
    input: DF(HashFunction, TransformedValues)
    output: DF(HashFunction, TransformedValues, StationarityResults, LinearityResults)
    """

    def apply_tests(self, transformed_df, feature_type):        
        transformed_df[feature_type] = transformed_df.TransformedValues.apply(lambda x : self.feature_data_per_block(x, feature_type))
        return transformed_df
        
        
    def feature_data_per_block(self, transformed_values, feature_type):
        feature_block_list = []

        for index in range(0, self.MINUTES_PER_DAY * self.number_of_days, self.block_size):
            feature_block_list.append(self.hypothesisTest(transformed_values[index : (index + self.block_size)], feature_type))

        return feature_block_list
        
    """
    Hypothesis testing for stationarity and linearity
    input: Block(List[Float])
    output: [1/0, 1/0, sum, int](List[Int])
    """

    def hypothesisTest(self, trace, feature_type):
        if min(trace) != max(trace):
            if feature_type == 'Stationarity':
                stat_res = ht.test_stationarity_adfuller(trace)
                if not math.isnan(stat_res):
                    if stat_res > 0.05:
                        stat_res = 0
                    elif stat_res <= 0.05:
                        stat_res = 1
                else:
                    stat_res = 1
                return stat_res
            elif feature_type == 'Linearity':
                lin_res = ht.test_non_linearity(trace)
                if not math.isnan(lin_res):
                    if lin_res < 0.05:
                        lin_res = 0
                    elif lin_res >= 0.05:
                        lin_res = 1
                else:
                    lin_res = 1
                return lin_res
            elif feature_type == 'Density':
                return sum(trace) / len(trace)
            elif feature_type == 'Harmonics':
                return ht.test_harmonics(trace, 1, 120, 2)
        else:
            if feature_type == 'Stationarity' or feature_type == 'Linearity':
                return 1
            elif feature_type == 'Density':
                return sum(trace) / self.block_size
            elif feature_type == 'Harmonics':
                return []


    """
    Applying multiprocessing to Hypothesis Testing
    input: [DF(HashFunction, TransformedValues), DF(HashFunction, TransformedValues), ...]
    output: DF(HashFunction, TransformedValues, BlockLists([Stat results, Lin results, Sum(Inv Conc)]))
    """

    def multiprocessor_apply_tests(self, transformed_df, feature_type):
        # separate into block columns
        transformed_dfs = np.array_split(transformed_df, self.max_workers)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            test_results = executor.map(self.apply_tests, transformed_dfs, [feature_type]*self.max_workers)
        
        result = pd.concat(test_results)

        result = result[["HashApp", feature_type]]
         
        return result

    def transform_func_to_app_level_ht(self, harmonics_df):
        harmonics_df = harmonics_df.groupby("HashApp", group_keys=False).agg({"HashApp": list, "Harmonics": list})

        harmonics_df["HashApp"] = harmonics_df.HashApp.apply(lambda x : x[0])
        harmonics_df["Harmonics"] = harmonics_df.Harmonics.apply(lambda x : combine_harmonics(x))
        
        return harmonics_df


def combine_harmonics(harmonics_per_func):
    """Combine func-level harmonics into application-level by adding all unique harmonics in each block to the
    application-level blocks. 

    Returns: list[int]
    number of different harmonics in each block for the application
    """

    num_blocks = len(harmonics_per_func[0])
    harmonics_list = [set([]) for _ in range(num_blocks)]
    
    for func_vals in harmonics_per_func:
        for block_index in range(num_blocks):
            for elem in func_vals[block_index]:
                harmonics_list[block_index].add(elem)

    return [len(block_harmonics) for block_harmonics in harmonics_list]
    

if __name__ == "__main__":
    dppd = HypothesisTesting(max_workers=48, block_size=504) 
    feature_types = ["Harmonics", "Density", "Stationarity", "Linearity"]

    FUNC_PATH = "../../data/azure/transformed_data/concurrency/func/"
    PATH = "../../data/azure/transformed_data/concurrency/" 
    
    dppd.process_transformed_traces(PATH, FUNC_PATH, feature_types)
