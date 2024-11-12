

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import functions from Data_Preprocessing and Models
import Data_Preprocessing as dp
import Models as mdl

# Load metadata and create datasets
metadata = dp.load_metadata()

# Prepare MFCC dataset for classical ML models
X_mfcc, y_mfcc = dp.create_mfcc_dataset(metadata)
X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(X_mfcc, y_mfcc, test_size=0.2, random_state=42)

# Train and evaluate SVM
svm_model = mdl.train_svm(X_train_mfcc, y_train_mfcc)
y_pred_svm = svm_model.predict(X_test_mfcc)
print("SVM Accuracy:", accuracy_score(y_test_mfcc, y_pred_svm))
print(classification_report(y_test_mfcc, y_pred_svm))

# Train and evaluate Random Forest
rf_model = mdl.train_random_forest(X_train_mfcc, y_train_mfcc)
y_pred_rf = rf_model.predict(X_test_mfcc)
print("Random Forest Accuracy:", accuracy_score(y_test_mfcc, y_pred_rf))
print(classification_report(y_test_mfcc, y_pred_rf))

# Prepare Mel Spectrogram dataset for CNN model
X_mel, y_mel = dp.create_mel_spectrogram_dataset(metadata)
y_mel_encoded, label_encoder = mdl.encode_labels(y_mel)
y_mel_encoded = to_categorical(y_mel_encoded)

X_train_mel, X_test_mel, y_train_mel, y_test_mel = train_test_split(X_mel, y_mel_encoded, test_size=0.2, random_state=42)

# Build and train CNN model
cnn_model = mdl.create_cnn_model(input_shape=(128, 128, 1), num_classes=y_train_mel.shape[1])
history = mdl.train_cnn(cnn_model, X_train_mel, y_train_mel)

# Evaluate CNN model
cnn_eval = cnn_model.evaluate(X_test_mel, y_test_mel, verbose=0)
print("CNN Accuracy:", cnn_eval[1])

# Predictions and Classification Report for CNN
y_pred_cnn = cnn_model.predict(X_test_mel)
y_pred_classes = np.argmax(y_pred_cnn, axis=1)
y_true_classes = np.argmax(y_test_mel, axis=1)
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))
