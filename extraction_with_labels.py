import os
import numpy as np
import torch
from tqdm import tqdm
import librosa
from Datasets import ESC50Dataset
from utils import pad_or_truncate

# Directories for the dataset and processed data
data_dir = r"C:\Users\jimmy\Desktop\Practical_Work\dataset"
processed_data_dir = "C:/Users/jimmy/Desktop/Practical_Work/processed_data/mel_spectrogram"

def save_mel_spectrogram_dataset(
    data_loader, split_name, save_dir, audio_length=160000,
    n_fft=2048, hop_length=512, win_length=None, n_mels=128, fmax=8000
    ):
    # Create split directory
    split_save_dir = os.path.join(save_dir, split_name)
    os.makedirs(split_save_dir, exist_ok=True)

    features, labels = [], []

    for i, (waveform, filename, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Handle shape depending on the input format
        if len(waveform.shape) == 3 and waveform.shape[1] == 1:  # [batch_size, 1, num_samples]
            waveform = waveform.squeeze(1)
        elif len(waveform.shape) == 2:  # [batch_size, num_samples]
            pass
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # Convert waveform to numpy and process
        waveform = waveform.numpy()

        for idx, audio in enumerate(waveform):
            # Pad or truncate to fixed length
            audio = pad_or_truncate(audio, audio_length)

            # Generate Mel Spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=32000,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                fmax=fmax
            )

            # Convert to log scale
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Save Mel Spectrogram
            feature_filename = os.path.join(split_save_dir, f"{filename[idx]}.npy")
            np.save(feature_filename, mel_spectrogram)

            # Append feature path and label for metadata
            features.append(feature_filename)
            labels.append(target[idx].numpy())

    # Save metadata
    np.save(os.path.join(split_save_dir, "labels.npy"), np.array(labels))
    np.save(os.path.join(split_save_dir, "features.npy"), np.array(features))
    print(f"{split_name} dataset saved in {split_save_dir}")


if __name__ == "__main__":
    os.makedirs(processed_data_dir, exist_ok=True)

    dataset_config = {
        'meta_csv': os.path.join(data_dir, "esc50.csv"),
        'audio_path': os.path.join(data_dir, "audio/"),
        'num_of_classes': 50,
    }

    # Create train and test datasets
    train_dataset = ESC50Dataset(
        metadata=dataset_config['meta_csv'],
        data_dir=dataset_config['audio_path'],
        fold=1,
        train=True
    )

    test_dataset = ESC50Dataset(
        metadata=dataset_config['meta_csv'],
        data_dir=dataset_config['audio_path'],
        fold=1,
        train=False
    )

    # Data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    save_mel_spectrogram_dataset(
        train_loader, "train", processed_data_dir,
        n_fft=2048, hop_length=512, win_length=1024, n_mels=128, fmax=8000
    )
    save_mel_spectrogram_dataset(
        test_loader, "test", processed_data_dir,
        n_fft=2048, hop_length=512, win_length=1024, n_mels=128, fmax=8000
    )