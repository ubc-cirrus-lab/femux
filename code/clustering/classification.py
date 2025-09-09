import pickle
import time
import warnings
import sys
import os
sys.path.append("../")
from clustering.decision_tree import train_decision_tree, test_decision_tree
from clustering.kmeans import train_kmeans, test_kmeans
from clustering.pre_processing import get_train_test_data
sys.path.append("../../plots/plotters/")
from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure" / "clustering") + "/"

warnings.filterwarnings('ignore')

train_dir = data_dir + 'training_output/'
test_dir = data_dir + 'testing_output/'
km_model_file = train_dir + 'training_kmeans_model_{}.pickle'
training_model_file = train_dir + 'training_model_{}_{}_{}_{}_{}.pickle'
kmeans_transformer_file = train_dir + 'training_transformer_{}_{}_{}_{}_{}.pickle'
km_transformer_file = train_dir + 'training_kmeans_transformer_{}.pickle'
km_all_data_transformer_file = train_dir + 'training_kmeans_global_transformer_{}.pickle'
classification_df_file = test_dir + 'testing_{}_{}_{}_{}_{}_percent_df_{}.pickle'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
    
def classification(classification_type, forecasters, features, percentage, transformer, block_size, weight_mode, num_workers):
    feature_names = "_".join(features)
    print("Running " + classification_df_file.format(classification_type, weight_mode, transformer, block_size, feature_names, percentage))
    print("with features {}".format(features))
     
    train_df, test_df = get_train_test_data(forecasters, percentage, block_size, weight_mode, features) 

    if classification_type == "kmeans":
        kmeans_model, transformer_model, forecaster_map = train_kmeans(train_df, features)
        
        with open(training_model_file.format(classification_type, weight_mode, transformer, block_size, feature_names), 'wb') as f:
            pickle.dump(kmeans_model, f)

        with open(kmeans_transformer_file.format(classification_type, weight_mode, transformer, block_size, feature_names), 'wb') as f:
            pickle.dump(transformer_model, f)

        test_df = test_kmeans(kmeans_model, transformer_model, test_df, forecaster_map, features)

        test_df.to_pickle(classification_df_file.format(classification_type, weight_mode, transformer, block_size, percentage, feature_names))


    elif classification_type == "decision_tree":
        dt_model, forecaster_map = train_decision_tree(train_df, features)
        
        with open(training_model_file.format(classification_type, weight_mode, transformer, block_size), 'wb') as f:
            pickle.dump(dt_model, f)

        test_df = test_decision_tree(dt_model, test_df, forecaster_map, features)

        test_df.to_pickle(classification_df_file.format(classification_type, weight_mode, transformer, block_size, percentage))

    else:
        raise Exception("unsupported model")


if __name__ == '__main__':
    percentage = 20
    forecasters = ["AR_10", "SETAR", "FFT_10", "Holt", "MarkovChain_v3", "ExpSmoothing", "10_min_keepalive", "5_min_keepalive"]
    transformer = "DecisionTree"
    features = ["Density", "Linearity", "Stationarity", "Harmonics"]
    block_size = 504
    weight_mode = "default"
    num_workers = 12
    
    classification("decision_tree", forecasters, features, percentage, transformer, block_size, weight_mode, num_workers)