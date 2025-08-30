import os
import numpy as np
import pandas as pd
from plotter import plot
from pathlib import Path

output_plots_dir = str(Path(__file__).parents[1] / "output_plots" / "azure") + "/"
data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"


cluster_file = data_dir + "clustering/testing_output/testing_kmeans_default_markov_v3_StandardScaler_100_percent_df.pickle"
save_path = output_plots_dir + "k_means_stats/"

os.makedirs(output_plots_dir + "k_means_stats/", exist_ok=True)

forecaster_shapes = ['o', 's']

ALL_FEATURES = ["Stationarity", "Density", "Linearity", "Harmonics"]

def plot_stats(features, mode):
    cluster_dfs = pd.read_pickle(cluster_file)

    cluster_df, feature_col_name = get_clustering_df(cluster_dfs, features)

    cluster_df.to_csv("check.csv")

    plot_cluster_stats(cluster_df, feature_col_name, features)
    plot_forecaster_stats(cluster_df, mode)


def plot_cluster_stats(cluster_df, feature_col_name, features):
    cluster_df["skip"] = cluster_df.apply(lambda x : remove_ties(x.AR_Metrics, x.Holt_Metrics), axis=1)
    print(cluster_df)
    cluster_df = cluster_df[cluster_df["skip"] != True]
    print(cluster_df)
    
    num_clusters = max(cluster_df[feature_col_name].unique())
    
    args = dict()

    cluster_winners = []
    for i in range(num_clusters):
        cluster_rows = cluster_df[cluster_df[feature_col_name] == i]
        if cluster_rows.empty:
            cluster_winners.append([None, 0])
        else:
            cluster_winners.append([cluster_rows["Cluster_Forecaster"].to_list()[0], len(cluster_rows)])
    
    print(cluster_winners)

    forecasters = ["SETAR", "AR", "Holt", "ExpSmoothing", "MarkovChain_v3", "FFT_10", "10_min_keepalive", "5_min_keepalive"]
    blocks_won = []

    for forecaster in forecasters:
        forecaster += "_Metrics"
        num_clusters_won = 0
        num_blocks_won = 0
        for cluster_winner in cluster_winners:
            if cluster_winner[0] == forecaster:
                num_clusters_won += 1
                num_blocks_won += cluster_winner[1]

        blocks_won.append(num_blocks_won) 
        print(forecaster)
        print(num_blocks_won)
        print(num_clusters_won)
    return
    df = pd.DataFrame({"Forecasters": forecasters, "NumBlockWon": blocks_won})

    args["y_label"] = ["{} ({} blocks)".format(cluster_winner[0], cluster_winner[1]) for cluster_winner in cluster_winners]
    args["kmeans_box"] = True
    args["title"] = "Test Split for 100% data"
    
    for feature in ALL_FEATURES:
        args["x_label"] = "Transformed {} Values".format(feature)
        args["file_name"] = save_path + "_".join(features) + "/" + feature 
        

        feature_vals = []

        print("Gather {} Values".format(feature))
        for cluster_id in range(num_clusters):            
            cur_df = cluster_df[cluster_df[feature_col_name] == cluster_id]
            feature_vals.append(cur_df[feature].to_list())

        print("Plotting {}".format(feature))
        plot(feature_vals, 99, args, close=True)


def plot_forecaster_stats(cluster_df, mode):
    cluster_df["skip"] = cluster_df.apply(lambda x : remove_ties(x.AR_Metrics, x.Holt_Metrics), axis=1)
    print(cluster_df)
    cluster_df = cluster_df[cluster_df["skip"] != True]
    print(cluster_df)
    forecasters = cluster_df["Oracle_Forecaster"].unique()

    args = dict()

    #args["kmeans_box"] = True
    args["title"] = "Test Split for 100% data"
    close = False
    
    for feature in ALL_FEATURES:
        args["y_label"] = "Fraction of Blocks"
        args["x_label"] = "Transformed {} Values".format(feature)
        args["file_name"] = save_path + "forecasters_" + feature + "_" + mode
        args["log"] = True if feature == "Density" or feature == "Harmonics" else False
        
        feature_vals = []

        print("Gather {} Values".format(feature))
        for i, forecaster in enumerate(forecasters):            
            cur_df = cluster_df[cluster_df["{}_Forecaster".format(mode)] == forecaster]
            block_vals = cur_df[feature].to_list()
            print(min(block_vals))
            print(max(block_vals))
            #args["y_label"].append("{} ({})".format(forecaster, len(block_vals)))
            args["label"] = "{} ({})".format(forecaster, len(block_vals))
            args["forecaster_num"] = i
            #feature_vals.append(block_vals)

            print("Plotting {}".format(feature))
            if i == len(forecasters) - 1:
                close = True
            plot(block_vals, 99, args, close=close)
        
        close = False


def remove_ties(AR_Metric, Holt_Metric):
    return np.isclose(AR_Metric, Holt_Metric)


def get_clustering_df(cluster_dfs, desired_features):
    """ Get the df of the clustering that was done based on the desired set of features
    cluster_dfs: list[pd.DataFrame]

    features: list[str]
    desired features ("Linearity", "Stationarity"...)
    """
    for df in cluster_dfs:
        feature_col_name = [col for col in df.columns if "Clusters" in col][0]
        features_match = True

        for feature in ALL_FEATURES:
            if feature in desired_features:
                if feature not in feature_col_name:
                    features_match = False
                    break
            # if feature is not desired then we want to make sure it's not included
            elif feature in feature_col_name:
                features_match = False
                break
        
        if features_match:
            return df, feature_col_name
    
    raise Exception("No df matches desired features")


if __name__ == "__main__":
    features = ["Density", "Linearity", "Stationarity", "Harmonics"]
    os.makedirs(output_plots_dir + "k_means_stats/" + "_".join(features) + "/", exist_ok=True)
    for mode in ["Cluster"]:
        plot_stats(features, mode)