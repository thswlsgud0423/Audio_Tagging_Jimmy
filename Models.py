# Models.py
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def train_svm(X_train, y_train):
    """Trains an SVM model."""
    svm_clf = SVC(kernel='linear')
    svm_clf.fit(X_train, y_train)
    return svm_clf

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train, y_train)
    return rf_clf

def create_cnn_model(input_shape, num_classes):
    """Creates and compiles a CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
    """Trains the CNN model."""
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def encode_labels(y):
    """Encodes labels as integers and converts to categorical."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return to_categorical(y_encoded), le  # Directly convert to categorical here
