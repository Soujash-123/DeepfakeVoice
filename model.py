import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# File paths
DATA_PATH = "./KAGGLE/DATASET-balanced.csv"
MODEL_PATH = "models/deepfake_model.keras"
OUTPUT_DIR = "outputs"
RESULTS_MD = os.path.join(OUTPUT_DIR, "results.md")

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Function to write to Markdown file
def write_to_md(content):
    with open(RESULTS_MD, "a") as f:
        f.write(content + "\n\n")

# Initialize Markdown file
with open(RESULTS_MD, "w") as f:
    f.write(f"# Deepfake Voice Detection Results\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# Load and preprocess data
try:
    df = pd.read_csv(DATA_PATH)
    write_to_md(f"Data loaded successfully from {DATA_PATH}")
except FileNotFoundError:
    error_msg = f"Error: The file {DATA_PATH} was not found. Please check the file path."
    print(error_msg)
    write_to_md(error_msg)
    exit(1)

# Label encoding
label_encoder = LabelEncoder()
df['LABEL'] = label_encoder.fit_transform(df['LABEL'])

# Split features and labels
y = df['LABEL']
X = df.drop('LABEL', axis=1)

# Normalize features
X = MinMaxScaler().fit_transform(X)

write_to_md(f"Data preprocessing completed. Shape of X: {X.shape}")

# Prepare data for LSTM
def prepare_data(X, window_size=5):
    data = []
    for i in range(len(X)):
        row = X[i]
        row_data = []
        for j in range(len(row) - window_size + 1):
            window = row[j:j + window_size]
            row_data.append(window)
        data.append(row_data)
    return np.array(data)

X = prepare_data(X)
write_to_md(f"Data prepared for LSTM. New shape of X: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=42)

write_to_md(f"Data split into train, validation, and test sets.")
write_to_md(f"Train set shape: {X_train.shape}")
write_to_md(f"Validation set shape: {X_val.shape}")
write_to_md(f"Test set shape: {X_test.shape}")

# Model architecture
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

write_to_md("## Model Architecture")
model.summary(print_fn=lambda x: write_to_md(x))

# Callbacks
callbacks = [
    ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# Train the model
try:
    write_to_md("## Training")
    history = model.fit(X_train, y_train, batch_size=32, epochs=60, validation_data=(X_val, y_val), callbacks=callbacks)
    write_to_md("Model training completed successfully.")
except Exception as e:
    error_msg = f"An error occurred during model training: {str(e)}"
    print(error_msg)
    write_to_md(error_msg)
    exit(1)

# Load the best model
try:
    model = load_model(MODEL_PATH)
    write_to_md(f"Best model loaded from {MODEL_PATH}")
except Exception as e:
    error_msg = f"An error occurred while loading the model: {str(e)}"
    print(error_msg)
    write_to_md(error_msg)
    exit(1)

# Predict
y_pred = np.round(model.predict(X_test).flatten())

# Evaluate
write_to_md("## Evaluation Metrics")
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred)
}

for metric, value in metrics.items():
    write_to_md(f"- **{metric}:** {value:.4f}")
    print(f"{metric}: {value:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='.4g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
confusion_matrix_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.close()
write_to_md(f"![Confusion Matrix]({confusion_matrix_path})")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
training_history_path = os.path.join(OUTPUT_DIR, 'training_history.png')
plt.savefig(training_history_path)
plt.close()
write_to_md(f"![Training History]({training_history_path})")

print(f"Evaluation complete. Results have been saved to {RESULTS_MD}")
