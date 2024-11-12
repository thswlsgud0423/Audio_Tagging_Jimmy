import os
import librosa
import numpy as np
import pandas as pd
import cv2

sample_rate = 22050
n_mfcc = 13

def load_metadata(file_path='C:\\Users\\jimmy\\Desktop\\Practical_Work\\ESC-50-master\\ESC-50-master\\meta\\esc50.csv'):
    return pd.read_csv(file_path)

# mfcc = Mel-frequency Cepstral Coefficients
def extract_mfcc_features(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return cv2.resize(log_mel_spec, (128, 128))

def create_mfcc_dataset(metadata, audio_path='C:\\Users\\jimmy\\Desktop\\Practical_Work\\ESC-50-master\\ESC-50-master\\audio\\'):
    """Creates a dataset of MFCC features and labels."""
    features, labels = [], []
    for _, row in metadata.iterrows():
        file_path = os.path.join(audio_path, row['filename'])
        if not os.path.isfile(file_path):  # Check if the file exists
            print(f"File not found: {file_path}")  # Print missing file path for debugging
            continue  # Skip to the next iteration if the file is missing
        mfcc = extract_mfcc_features(file_path)
        features.append(mfcc)
        labels.append(row['category'])
    return np.array(features), np.array(labels)

def create_mel_spectrogram_dataset(metadata, audio_path='C:\\Users\\jimmy\\Desktop\\Practical_Work\\ESC-50-master\\ESC-50-master\\audio\\'):
    """Creates a dataset of Mel Spectrogram features and labels."""
    features, labels = [], []
    for _, row in metadata.iterrows():
        file_path = os.path.join(audio_path, row['filename'])
        if not os.path.isfile(file_path):  # Check if the file exists
            print(f"File not found: {file_path}")  # Print missing file path for debugging
            continue  # Skip to the next iteration if the file is missing
        mel_spec = extract_mel_spectrogram(file_path)
        features.append(mel_spec)
        labels.append(row['category'])
    features = np.array(features).reshape(-1, 128, 128, 1)
    return features, np.array(labels)
