import argparse
import sys
from pathlib import Path
from time import time

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))
import data
from models.encoder_decoder_dropout import *

import utils


def train_lstm(n_input_steps, n_output_steps, samples, 
         model_artifacts_dir, num_epochs, batch_size, 
         learning_rate, variational_dropout_p, hashapp):
    # --------------------------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------------------------
    datasets = data.get_datasets(
        samples=samples, n_input_steps=n_input_steps, pretraining=True
    )
    
    encoder_in_features = datasets["train"].X.shape[-1]  # 5
    device = utils.get_device()

    # --------------------------------------------------------------------------
    # Train LSTM encoder decoder
    # --------------------------------------------------------------------------
    model = VDEncoderDecoder(
        in_features=encoder_in_features,
        input_steps=n_input_steps,
        output_steps=n_output_steps,
        p=variational_dropout_p,
    )

    start = time()
    model, losses = utils.train_encoder_decoder(
        device=device,
        model=model,
        datasets=datasets,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_tqdm=False,
    )
    print("Training took: {:.2f} minutes".format((time() - start) / 60))
    
    utils.save(model, name="lstm_encoder_decoder_{}".format(hashapp), path=model_artifacts_dir)


if __name__ == "__main__":
    train_lstm()
