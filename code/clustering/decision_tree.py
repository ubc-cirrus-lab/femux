from sklearn import tree, ensemble
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = str(Path(__file__).parents[2] / "data" / "azure" / "clustering") + "/"

test_dir = data_dir + 'testing_output/'

classification_df_file = test_dir + 'testing_{}_{}_{}_{}_{}percent_df.pickle'

def train_decision_tree(df, features):
    forecaster_map, forecaster_cols = map_forecaster_to_int(df)

    df["Oracle_Forecaster"] = df[forecaster_cols].apply(lambda x: np.argmin(x.values.tolist()), axis=1)

    # give even columns to train and odd for validation, this is to split the small, medium, and large data between them
    train_df = df.iloc[::2]
    valid_df = df.iloc[1::2]
    
    max_depths = [3, 6, 9, 15, 19, 30, None]

    best_validation_score = 0

    for max_depths in max_depths:
        clf = tree.DecisionTreeClassifier(max_depth=max_depths, criterion="entropy")
        clf = clf.fit(train_df[features], train_df["Oracle_Forecaster"])

        train_accuracy = clf.score(train_df[features], train_df["Oracle_Forecaster"])
        validation_accuracy = clf.score(valid_df[features], valid_df["Oracle_Forecaster"])

        if validation_accuracy > best_validation_score:
            best_validation_score = validation_accuracy
            best_clf = clf

    return best_clf, forecaster_map

def test_decision_tree(clf, df, forecaster_map, features):
    df["Oracle_Forecaster"] = df[forecaster_map.values()].apply(lambda x: np.argmin(x.values.tolist()), axis=1)
    print("Test score: ", clf.score(df[features], df["Oracle_Forecaster"]))
    
    df["Cluster_Forecaster"] = clf.predict(df[features])
    df["Cluster_Forecaster"] = df["Cluster_Forecaster"].apply(lambda x: forecaster_map[x])
    df["Oracle_Forecaster"] = df["Oracle_Forecaster"].apply(lambda x: forecaster_map[x])
    
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
    #feature_set = list(chain.from_iterable(combinations(feature_set, r) for r in range(len(feature_set)+1)))[1:]
    #print(feature_set)

    percentage = 20
    forecasters = ["AR_10", "SETAR", "FFT_10", "Holt", "MarkovChain_v3", "ExpSmoothing", "10_min_keepalive", "5_min_keepalive"]
    transformer = None
    classification_type = "decision_tree"
    block_size = 504
    weight_mode = "default"

    dt_model, forecaster_map = train_decision_tree(train_df, features)
    #with open(dt_model_file.format(file_description), 'wb') as f:
    #    pickle.dump(dt_model, f)

    test_df = test_decision_tree(dt_model, test_df, forecaster_map, features)
        
    test_df.to_pickle(classification_df_file.format(classification_type, weight_mode, transformer, block_size, percentage))

