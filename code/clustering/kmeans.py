from sklearn import tree, ensemble
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data_dir = str(Path(__file__).parents[2] / "data" / "azure" / "clustering") + "/"

test_dir = data_dir + 'testing_output/'

classification_df_file = test_dir + 'testing_{}_{}_{}_{}_{}percent_df.pickle'

def train_kmeans(df, features):
    forecaster_map, forecaster_cols = map_forecaster_to_int(df)

    # give even columns to train and odd for validation, this is to split the small, medium, and large data between them
    train_df = df.iloc[::2]
    valid_df = df.iloc[1::2]
    
    transformer = StandardScaler()
    transformed_df = transformer.fit_transform(train_df[features])
    train_df[features] = transformed_df
    valid_df[features] = transformer.transform(valid_df[features])

    
    k_values = [10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    best_training_RUM = 1e20

    for k_value in k_values:
        kmeans = KMeans(n_clusters=k_value).fit(train_df[features])
        kmeans_labels = kmeans.predict(valid_df[features])
        valid_df["Cluster"] = kmeans_labels

        cur_rum = 0

        # for each cluster, get the minimum RUM across all forecasters
        # we don't care which forecaster it is right now.
        for i in range(k_value):
            cur_df = valid_df[valid_df.Cluster == i]
            cur_rum += min(cur_df[forecaster_cols].sum())
            
        if cur_rum < best_training_RUM:
            best_training_RUM = cur_rum
            best_kmeans = kmeans

    return best_kmeans, transformer, forecaster_map


def test_kmeans(mod, transformer, df, forecaster_map, features):
    df[features] = transformer.transform(df[features])
    kmeans_labels = mod.predict(df[features])
    df["Cluster"] = kmeans_labels

    cluster_map = {}
    for i in range(max(kmeans_labels) + 1):
        cur_df = df[kmeans_labels == i]
        cluster_forecaster = cur_df[forecaster_map.values()].sum().idxmin()
        cluster_map[i] = cluster_forecaster

    df["Cluster_Forecaster"] = df["Cluster"].apply(lambda x: cluster_map[x])
    df["Oracle_Forecaster"] = df[forecaster_map.values()].apply(lambda x: forecaster_map[np.argmin(x.values.tolist())], axis=1)
    
    return df


def map_forecaster_to_int(df):
    forecaster_map = {}
    forecaster_cols = []
    i = 0

    for col_name in df.columns:
        if "Metrics" in col_name:
            forecaster_map[i] = col_name
            forecaster_cols.append(col_name)
            i += 1

    return forecaster_map, forecaster_cols
    

if __name__ == "__main__":
    train_df = pd.read_pickle("train.pickle")
    test_df = pd.read_pickle("test.pickle")
    features = ["Density", "Linearity", "Stationarity", "Harmonics"]

    percentage = 20
    forecasters = ["AR", "SETAR", "FFT_10", "Holt", "MarkovChain_v3", "ExpSmoothing", "10_min_keepalive", "5_min_keepalive"]
    transformer = None
    classification_type = "decision_tree"
    block_size = 504
    weight_mode = "default"

    kmeans_model, transformer, forecaster_map = train_kmeans(train_df, features)

    test_df = test_kmeans(kmeans_model, transformer, test_df, forecaster_map, features)

    print(test_df[["HashApp", "BlockIndex"]].head(50))