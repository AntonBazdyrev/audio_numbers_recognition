import os
import click

from torch import nn, optim, jit
from torch.utils.data import DataLoader
from catalyst import dl, utils

from utils import load_train_dataset, cer_callback
from model import create_model


@click.command()
@click.option('--dir_path', default='train_data/numbers2/', help='path to all training data')
@click.option('--train_filename', default='train.csv', help='name of train df in dir_path')
@click.option('--logdir', default='./train_logs', help='logdir for catalyst logs')
@click.option('--model_path', default='final_model.pt', help='name of the final model file')
def train(dir_path, train_filename, logdir, model_path):
    train_ds, valid_ds = load_train_dataset(dir_path, train_filename)

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loaders = {
        "train": DataLoader(
            train_ds, batch_size=32, shuffle=True
        ),
        "valid": DataLoader(
            valid_ds, batch_size=256, shuffle=False
        )
    }

    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=20,
        callbacks=[
            cer_callback
        ],
        logdir=logdir,
        valid_loader="valid",
        valid_metric="cer_metric",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )
    model = model.eval().cpu()
    scripted_model = jit.script(model)
    scripted_model.save(model_path)
    
if __name__ == '__main__':
    train()
