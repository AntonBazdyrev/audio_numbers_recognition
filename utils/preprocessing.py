import torch
import torchaudio
import numpy as np
import pandas as pd

from tqdm import tqdm
from os.path import join as path_join
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, ClippingDistortion


augmentation = Compose([
    AddGaussianNoise(min_amplitude=0.003, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ClippingDistortion(max_percentile_threshold=20, p=0.5)
])

def convert_single_wav(path, maxlen=int(460*1.2), n_mels=64, augmented=False):
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0).unsqueeze(0) #multitrack audio -> mono
    if augmented:
        waveform = augmentation(samples=waveform.numpy(), sample_rate=sample_rate)
        waveform = torch.FloatTensor(waveform)
    mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels)(waveform) 
    mel_specgram_db = torchaudio.transforms.AmplitudeToDB()(mel_specgram)
    if mel_specgram_db.shape[2] < maxlen:
        mel_specgram_db = torch.nn.functional.pad(
        mel_specgram_db, (maxlen - mel_specgram_db.shape[2], 0), mode='constant', value=0
    )
    else:
        mel_specgram_db = mel_specgram_db[:, :, :maxlen]
    return mel_specgram_db

def convert_dataset(dir_path, df, augmented=0):
    inputs = []
    targets = []
    for ind, row in tqdm(df.iterrows(), total=df.shape[0]):
        target = torch.LongTensor([int(c) for c in row.target])
        targets.append(target)
        inputs.append(convert_single_wav(path_join(dir_path, row.path)))
        for i in range(augmented):
            inputs.append(convert_single_wav(path_join(dir_path, row.path), augmented=True))
            targets.append(target)
    ds = TensorDataset(torch.stack(inputs), torch.stack(targets))
    return ds

def load_train_dataset(dir_path, filename, test_size=0.3):
    df = pd.read_csv(path_join(dir_path, filename))
    df = df[df.number.notna()].reset_index(drop=True)
    df['target'] = df.number.apply(int).apply(str).apply(lambda x: ((6 - len(x))*'0' + x))
    df_train, df_valid = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
    train = convert_dataset(dir_path, df_train, augmented=2)
    valid = convert_dataset(dir_path, df_valid)
    return train, valid

def load_inference_dataset(dir_path, filename):
    df = pd.read_csv(path_join(dir_path, filename))
    inputs = []
    for path in df.path:
        inputs.append(convert_single_wav(path_join(dir_path, path)))
    ds = TensorDataset(torch.stack(inputs))
    return ds
