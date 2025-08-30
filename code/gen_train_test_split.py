import pandas as pd
import numpy as np
import pickle

data_path = "../data/azure/transformed_data/concurrency/app/{}_app_conc_00.pickle"
train_save_path = "../data/azure/train_test_split/{}_training_apps.pickle"
test_save_path = "../data/azure/train_test_split/{}_test_apps.pickle"


def gen_train_split_hashapps(percentage):
    """Generate and store a representative sample of applications used 
    for training

    percentage: int
    Training split size
    """
    num_files = {"small": 40, "medium": 4, "large": 1}

    for size in ["small", "medium", "large"]:
        cur_path = data_path.format(size)
        hashapp_list = []
        
        for file_index in range(num_files[size]):
            df = pd.read_pickle(cur_path.replace("_00", "_{:02d}".format(file_index)))
            hashapp_list.extend(df.HashApp.to_list())

        num_training_apps = int(np.ceil(len(hashapp_list) * 0.7))
        print(num_training_apps)
        
        with open(train_save_path.format(size), 'wb') as f:
            pickle.dump(hashapp_list[:num_training_apps], f)

        with open(test_save_path.format(size), 'wb') as f:
            pickle.dump(hashapp_list[num_training_apps:], f)


gen_train_split_hashapps(70)