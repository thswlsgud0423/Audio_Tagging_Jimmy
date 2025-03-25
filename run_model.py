import os
from Datasets import ESC50Dataset
import torch 
import numpy as np
from mel_spectrogram_extraction import save_mel_spectrogram_dataset
# The pretrained model is from https://github.com/fschmid56/EfficientAT
from models.mn.model import get_model as get_mn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import train_model, evaluate_model
from Datasets import MelSpectrogramDataset
# Models
from models.dymn.model import get_model as get_dymn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import train_model, evaluate_model
from Datasets import MelSpectrogramDataset
# Models
from Models import CRNN

transform = transforms.Compose([
    transforms.Lambda(lambda x: (x - np.mean(x)) / (np.std(x) + 1e-6))  # Normalize Mel spectrogram
])


pretrained_model = get_mn(pretrained_name="mn10_as")

if __name__ == "__main__":
    processed_data_dir = "custom_processed_dataset"
    os.makedirs(processed_data_dir, exist_ok=True)

    dataset_config = {
        'meta_csv': os.path.join("dataset", "esc50.csv"),
        'audio_path': os.path.join("dataset", "audio"),
        'num_of_classes': 50,
    }

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    mel_config = {
        'audio_length': 160000,
        'n_fft': 1024,
        'hop_length': 320,
        'win_length': 320,
        'n_mels': 128,
        'fmax': None,
    }

    save_mel_spectrogram_dataset(train_loader, "train", processed_data_dir, mel_config)
    save_mel_spectrogram_dataset(test_loader, "test", processed_data_dir, mel_config)

custom_train_dir = "C:/Users/jimmy/Desktop/Practical_Work/custom_processed_dataset/train"
custom_test_dir = "C:/Users/jimmy/Desktop/Practical_Work/custom_processed_dataset/test"

custom_train_dataset = MelSpectrogramDataset(
    features_dir=custom_train_dir,
    labels_path=os.path.join(custom_train_dir, "labels.npy"),
    transform=transform
)

custom_test_dataset = MelSpectrogramDataset(
    features_dir=custom_test_dir,
    labels_path=os.path.join(custom_test_dir, "labels.npy"),
    transform=transform
)

# DataLoaders for train and test datasets
batch_size = 32
custom_train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True)
custom_test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Ensure the model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)

# Get a sample from the DataLoader
custom_sample_input, custom_sample_label = next(iter(custom_train_loader))

# Move the sample input and label to the device
custom_sample_input = custom_sample_input.to(device)
custom_sample_label = custom_sample_label.to(device)

# Perform the forward pass
output = pretrained_model(custom_sample_input)

loss = torch.nn.CrossEntropyLoss()
loss.forward(input = output[0], target = custom_sample_label)

test_accuracy = evaluate_model(pretrained_model, custom_test_loader, device)

def train_model(model, train_loader, test_loader, device, num_epochs=25):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_accuracy = (correct_predictions / total_predictions) * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print("Training complete!")

train_model(model=pretrained_model,
            train_loader=custom_train_loader,
            test_loader=custom_test_loader,
            device=device,
            num_epochs=20)

test_accuracy = evaluate_model(pretrained_model, custom_test_loader, device)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")