import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras_tuner as kt  # Keras Tuner
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load a single .ts file
def load_data(file_path):
    ts_data = pd.read_csv(file_path, delimiter='\t', header=None)
    expected_columns = [
        'Elapsed Time', 'Left Stride Interval', 'Right Stride Interval',
        'Left Swing Interval', 'Right Swing Interval', 'Left Stance Interval',
        'Right Stance Interval', 'Left Foot Strike', 'Right Foot Strike',
        'Extra Col 1', 'Extra Col 2', 'Extra Col 3', 'Extra Col 4'
    ]
    ts_data.columns = expected_columns[:ts_data.shape[1]]  # Adjust to match actual columns
    return ts_data

# Step 2: Load multiple .ts files from a directory
def load_multiple_files(directory_path):
    all_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.ts'):
            file_path = os.path.join(directory_path, file_name)
            ts_data = load_data(file_path)
            all_data.append(ts_data)
    return all_data

# Step 3: Preprocess Data - Padding and One-hot encoding labels
def preprocess_data(all_data):
    X = []
    y = []
    for ts_data in all_data:
        features = ts_data[['Left Stride Interval', 'Right Stride Interval', 'Left Swing Interval', 'Right Swing Interval']].values
        label = ts_data['Left Foot Strike'].values[-1]  # Assuming 'Left Foot Strike' is the label for classification
        X.append(features)
        y.append(label)

    # Pad sequences to handle varying lengths
    X = pad_sequences(X, padding='post', dtype='float32')
    y = np.array(y)

    # One-hot encode labels for classification
    y = to_categorical(y, num_classes=2)  # Assuming binary classification
    return np.array(X), y

# Step 4: Build a simple model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Train the model and plot visualizations
def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    model = build_model(X_train.shape[1:])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

    # Plot training & validation accuracy over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot training & validation loss over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    return model

# Step 6: Evaluate model and generate confusion matrix, ROC, and Precision-Recall curves
def evaluate_model(model, X_val, y_val):
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # Assuming 0 = class 0, 1 = class 1
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])  # Using class 1 for the positive class
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

# Load your dataset and preprocess
directory_path = '/content/physionet.org/files/gaitndd/1.0.0'
all_data = load_multiple_files(directory_path)
X, y = preprocess_data(all_data)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
model = train_and_evaluate_model(X_train, y_train, X_val, y_val)

# Evaluate the model
evaluate_model(model, X_val, y_val)
