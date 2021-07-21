import torch
import click
import pandas as pd

from os.path import join as path_join
from utils import load_inference_dataset, convert_to_str


@click.command()
@click.option('--dir_path', default='train_data/numbers2/', help='path to all training data')
@click.option('--filename', default='test-example.csv', help='name of train df in dir_path')
@click.option('--model_path', default='final_model.pt', help='name of the final model file')
def inference(dir_path, filename, model_path):
    ds = load_inference_dataset(dir_path, filename)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    model = torch.jit.load(model_path)
    with torch.no_grad():
        predicts = []
        for x in dl:
            y_pred = model(x[0])
            predicts.append(y_pred.detach().cpu())
        predicts = torch.cat(predicts).argmax(axis=1).numpy()

    mod_predicts = [int(convert_to_str(p)) for p in predicts]
    df = pd.read_csv(path_join(dir_path, filename))
    df['number'] = mod_predicts
    df.to_csv(path_join(dir_path, 'results.csv'), index=False)

    
if __name__ == '__main__':
    inference()
