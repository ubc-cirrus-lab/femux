import sys, os
import data
import torch
import pandas as pd
import numpy as np
import models.variational_dropout as vd
from collections import deque
from train_lstm_encoder_decoder import train_lstm
from train_prediction_network import train_network
from concurrent.futures import ProcessPoolExecutor
from full_inference import inference
from models.predict import *
from time import time

from pathlib import Path
sys.path.append("..")

from results import utils

data_dir = str(Path(__file__).parents[2] / "data" / "azure") + "/"
os.makedirs("model_artifacts", exist_ok=True)
os.makedirs(data_dir + "forecaster_data/invocations", exist_ok=True)

preproc_path = data_dir + "preproc_data/app_total_inv_exec_{}_days.pickle"
forecast_path = data_dir + "forecaster_data/invocations/aquatope_forecasts_{}.pickle"

TOTAL_NUM_DAYS = 12
MINUTES_PER_DAY = 1440

INPUT_STEPS = 48
OUTPUT_STEPS = 1
NUM_TRAINING_DAYS = 7
NUM_EPOCHS = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
DROPOUT_P = 0.25

def main(split):    
    # Grab invocations per min, filter test dataset
    df = get_invocations(split)
    
    df[["TrainingMu", "TrainingStd"]] = df.apply(lambda x : train_model(x.InvocationsPerMin, x.HashApp), axis=1)

    df["ForecastedValues"] = df.apply(lambda x : gen_forecasts(x.InvocationsPerMin[NUM_TRAINING_DAYS * MINUTES_PER_DAY:], 
                                           x.HashApp, x.TrainingMu, x.TrainingStd), axis=1)
    
    df[["HashApp", "ForecastedValues"]].to_pickle(forecast_path.format(split))


def train_model(invocations, hashapp):
    _, _, samples = data.pipeline(
        n_input_steps=INPUT_STEPS,
        n_pred_steps=OUTPUT_STEPS,
        invocations=invocations[:NUM_TRAINING_DAYS * MINUTES_PER_DAY],
        num_days=NUM_TRAINING_DAYS)
    
    training_mu = samples["train"][:, 0, 0].mean()
    training_std = samples["train"][:, 0, 0].std()

    if not os.path.exists("model_artifacts/lstm_encoder_decoder_{}.pt".format(hashapp)):
        train_lstm(n_input_steps=INPUT_STEPS, n_output_steps=OUTPUT_STEPS, samples=samples,
                model_artifacts_dir="model_artifacts", num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, variational_dropout_p=DROPOUT_P, hashapp=hashapp)

        train_network(n_input_steps=INPUT_STEPS, n_output_steps=OUTPUT_STEPS, samples=samples,
                    model_artifacts_dir="model_artifacts", num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE, hashapp=hashapp)    

    return pd.Series([training_mu, training_std])


def gen_forecasts(invocations, hashapption, training_mu, training_sigma):
    df = data.load_dataset(invocations, num_days=TOTAL_NUM_DAYS - NUM_TRAINING_DAYS)

    e_1 = df.hour_of_day_sin.tolist()
    e_2 = df.hour_of_day_cos.tolist()
    e_3 = df.day_of_week_sin.tolist()
    e_4 = df.day_of_week_cos.tolist()

    # normalize invocations since we trained on normalized data
    input_data = [[(invocations[i] - training_mu) / training_sigma, e_1[i], e_2[i], e_3[i], e_4[i]] for i in range(len(invocations))]
    forecasts = np.zeros(len(invocations) - INPUT_STEPS)
    cur_window = deque(input_data[:INPUT_STEPS])
    model = load_model(hashapption)
    
    for cur_minute in range(INPUT_STEPS, len(invocations)):
        if min(cur_window)  == max(cur_window):
            cur_window.popleft()
            cur_window.append(input_data[cur_minute])
            continue

        external = [e_1[cur_minute], e_2[cur_minute], e_3[cur_minute], e_4[cur_minute]]
        forecast, _ = inference(cur_window, external, model=model, device="cpu", batch_size=BATCH_SIZE)
        
        # denormalize forecast
        forecasts[cur_minute - INPUT_STEPS] = forecast * training_sigma + training_mu
        cur_window.popleft() 
        cur_window.append(input_data[cur_minute])

    return forecasts


def dropout_on(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.train()


def load_model(hashapp):
    predict_loc = "model_artifacts/predict_{}.pt".format(hashapp)
    predict = torch.load(predict_loc, map_location="cpu").eval()
    model = predict.to("cpu")
    model = model.apply(dropout_on)
    return model


def get_invocations(split):
    """Get the invocations per minute for the test dataset at an application level (across functions).
    split: int, which 10% split of the test dataset to use
    """
    test_hashapps = utils.init_df("test", 100).HashApp.tolist()
    
    inv_df = pd.read_pickle(preproc_path.format(TOTAL_NUM_DAYS))
    inv_df = inv_df[inv_df.HashApp.isin(test_hashapps)]
    inv_df = inv_df[["HashApp", "InvocationsPerMin"]]
    inv_df = np.array_split(inv_df, 100)[split]

    return inv_df


if __name__ == "__main__":
    splits = [12, 19]
    for split in splits:
        main(split=split)

