from EfficientAT.models.mn.model import get_model as get_mn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import transforms
from utils import train_model, evaluate_model
from Datasets import MelSpectrogramDataset
# Models

from EfficientAT.models.dymn.model import get_model as get_dymn


transform = transforms.Compose([
    transforms.Lambda(lambda x: (x - np.mean(x)) / (np.std(x) + 1e-6))  # Normalize Mel spectrogram
])

train_dir = "C:/Users/jimmy/Desktop/Practical_Work/processed_data/mel_spectrogram/train"
test_dir = "C:/Users/jimmy/Desktop/Practical_Work/processed_data/mel_spectrogram/test"

train_dataset = MelSpectrogramDataset(
    features_dir=train_dir,
    labels_path=os.path.join(train_dir, "labels.npy"),
    transform=transform
)

test_dataset = MelSpectrogramDataset(
    features_dir=test_dir,
    labels_path=os.path.join(test_dir, "labels.npy"),
    transform=transform
)

# DataLoaders for train and test datasets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = get_mn(pretrained_name="mn10_as")

# Train the model
train_model(model, train_loader, test_loader, device, num_epochs=100)

# Evaluate the model on the test set
test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")