import sys
import os
import gc
from time import time
from itertools import chain, combinations

from clustering.classification import classification
from clustering.post_processing import post_process

from results.gen_results import gen_results

sys.path.append("..")

def cluster_pipeline():
    forecast_len = 1
    forecast_window = 120
    data_percentage = 100
    forecasters = ["AR", "FFT_10", "MarkovChain", "10_min_keepalive", "5_min_keepalive"] #("Holt", "ExpSmoothing", "SETAR")
    features = ["Density", "Linearity", "Stationarity", "Harmonics"]
    num_workers = 48
    block_size = 504
    transformer = "StandardScaler"
    weight_mode = "default_exec"
    classifier = "kmeans"

    for weight_mode in ["default_exec"]:
        print("Using forecasters: {}".format(forecasters))

        start_time = time()
        classification(classifier, forecasters, features, data_percentage, transformer, block_size, weight_mode, num_workers)
        print("Classification with {} took {}".format(classifier, time()-start_time))
        
        feature_names = "_".join(features)

        ## use clusters to build FeMux forecasts
        post_process(False, classifier, "10_min_keepalive", block_size=block_size, forecast_window=120, 
                    forecast_len=1, num_workers=num_workers, features=features, 
                    percentage=data_percentage, transformer=transformer, weight_mode=weight_mode)
                
        data_mode = "test"
        data_subscript = "{}_{}_percent_{}".format(block_size, data_percentage, data_mode)
         
        postproc_path = "processed_testing_df_{}_{}_{}_{}_{}_percent_{}.pickle".format(classifier, weight_mode, transformer, block_size, data_percentage, feature_names)       
        femux_name = "{}_{}_{}_femux_{}".format(weight_mode, classifier, transformer, feature_names) 
        
        gen_results(data_subscript, femux_name, forecast_window, forecast_len, num_workers, 
        data_mode, data_percentage, block_size, postproc_path)
        gc.collect()
                    

def powerset(features):
    return list(chain.from_iterable(combinations(features, r) for r in range(len(features)+1)))[1:]


if __name__ == "__main__":
    cluster_pipeline()