import pandas as pd
import numpy as np


def get_percentile_rows(percentile_list, data_filepath, save_filepath):
    """Get the rows that most closely match the number of invocations from the given percentile
    for a given day.

    percentile_list: list[int]
    each desired percentile for number of invocations
    """
    data_df = pd.read_pickle(data_filepath)

    num_list = data_df.NumInvocations.tolist()

    takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))

    row_indices = []

    for percentile in percentile_list:
        num = np.percentile(num_list, percentile)
        closest_val = takeClosest(num, num_list)
        row_indices.append(num_list.index(closest_val))

    percentiles_df = data_df.iloc[row_indices]

    percentiles_df.reset_index(drop=True, inplace=True)

    print(percentiles_df)
    percentiles_df.to_pickle(save_filepath)

data_filepath = "../data/preproc_data/invocation_data/preprocessed_data_00.pckl"
save_filepath = "../transform/percentiles.pckl"
get_percentile_rows([50,90,95,99], data_filepath, save_filepath)