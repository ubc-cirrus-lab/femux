import argparse
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

import data
from models.predict import *

import utils


def train_network(n_input_steps, n_output_steps, samples,
          model_artifacts_dir, num_epochs, batch_size, learning_rate, hashapp):
    # --------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------- 

    datasets = data.get_datasets(
        samples=samples, n_input_steps=n_input_steps, pretraining=False
    )

    # --------------------------------------------------------------------------
    # Train LSTM encoder decoder
    # --------------------------------------------------------------------------
    device = utils.get_device()
    encoder_decoder_loc = model_artifacts_dir + "/" + "lstm_encoder_decoder_{}.pt".format(hashapp)
    encoder_decoder = torch.load(encoder_decoder_loc)
    prediction_network = Predict(
        n_extracted_features=n_input_steps,
        n_external_features=4,
        n_output_steps=n_output_steps,
        p=0.2,
        encoder_decoder=encoder_decoder,
    )

    start = time()
    model, losses = utils.train_prediction_network(
        device=device,
        datasets=datasets,
        prediction_network=prediction_network,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_tqdm=False,
    )
    print("Training took: {:.2f} minutes".format((time() - start) / 60))

    utils.save(model, name="predict_{}".format(hashapp), path=model_artifacts_dir)


if __name__ == "__main__":
    train_network()
