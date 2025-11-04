# train_voice_model.py
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_DIR = "data/voice_dataset"
OUTPUT_CSV = "data/voice_features.csv"
MODEL_PATH = "models/voice_recognizer.pkl"

# ===== Ekstraksi MFCC =====
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    return mfcc

# ===== Loop dataset =====
data = []
for user in os.listdir(DATASET_DIR):
    user_folder = os.path.join(DATASET_DIR, user)
    if not os.path.isdir(user_folder):
        continue
    for file in os.listdir(user_folder):
        if file.endswith(".wav"):
            path = os.path.join(user_folder, file)
            features = extract_features(path)
            data.append([user] + list(features))

# ===== Simpan ke CSV =====
columns = ["label"] + [f"mfcc_{i}" for i in range(1, 21)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f" Dataset suara disimpan di {OUTPUT_CSV}")

# ===== Latih model =====
X = df.drop(columns=["label"])
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model voice recognizer berhasil dilatih!")

joblib.dump(model, MODEL_PATH)
print(f" Model disimpan ke {MODEL_PATH}")
