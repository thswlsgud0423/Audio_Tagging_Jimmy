import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import librosa
import pandas as pd

from utils import pad_or_truncate


# Mel Spectrograms Dataset
class MelSpectrogramDataset(Dataset):
    def __init__(self, features_dir, labels_path, transform=None):
        self.features_dir = features_dir
        self.labels = np.load(labels_path)
        self.features = np.load(os.path.join(features_dir, "features.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_path = self.features[idx]
        feature = np.load(feature_path)  # Mel Spectrogram
        label = self.labels[idx]  # Load corresponding label

        if self.transform:
            feature = self.transform(feature)

        # Mel spectrograms need to have a channel dimension for PyTorch Conv2D
        feature = np.expand_dims(feature, axis=0)

        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class ESC50Dataset(Dataset):
    def __init__(self, metadata, data_dir, fold, train=True, audio_length=160000, sr=32000):
        self.metadata = pd.read_csv(metadata)
        self.data_dir = data_dir
        self.audio_length = audio_length
        self.sr = sr
        self.train = train
        
        # Create a mapping for labels
        self.label_map = {label: idx for idx, label in enumerate(self.metadata['category'].unique())}
        
        # Filter data by fold
        self.data = self.metadata[self.metadata['fold'] != fold] if train else self.metadata[self.metadata['fold'] == fold]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        label = self.label_map[row['category']] # numerical value
        filepath = os.path.join(self.data_dir, filename)

        # Load .wav file
        waveform, _ = librosa.load(filepath, sr=self.sr)
        
        # Pad or truncate
        waveform = pad_or_truncate(waveform, self.audio_length)
        
        return torch.tensor(waveform, dtype=torch.float32), filename, torch.tensor(label, dtype=torch.long)
