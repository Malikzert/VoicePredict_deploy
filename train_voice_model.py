# ===========================================
# train_voice_model.py (Gradient Boosting + fitur stabil)
# ===========================================
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ============================
# Konfigurasi Path
# ============================
DATASET_DIR = "data/voice_dataset"
OUTPUT_CSV = "data/voice_features.csv"
MODEL_PATH = "models/voice_recognizer.pkl"

# ============================
# Ekstraksi Fitur MFCC (lebih kaya)
# ============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y)            # hilangkan jeda diam
    y = librosa.util.normalize(y)             # normalisasi volume

    # MFCC utama
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    # Delta MFCC (perubahan antar frame)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    # Gabungkan semua (60 dimensi total)
    features = np.concatenate([mfcc_mean, mfcc_var, mfcc_delta_mean])
    return features


# ============================
# Looping Dataset & Ekstraksi
# ============================
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

# ============================
# Simpan Dataset ke CSV
# ============================
columns = ["label"] + [f"feat_{i}" for i in range(1, 61)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Dataset suara disimpan di {OUTPUT_CSV}")

# ============================
# Siapkan Data untuk Training
# ============================
X = df.drop(columns=["label"])
y = df["label"]

# Normalisasi fitur agar distribusi stabil antar user
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Latih Model Gradient Boosting
# ============================
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    random_state=42
)
model.fit(X_scaled, y)
print("✅ Model voice recognizer (GradientBoostingClassifier) berhasil dilatih!")

# Simpan Model + Scaler
joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
print(f"💾 Model disimpan ke {MODEL_PATH}")

# ============================
# 🔍 Evaluasi Confidence dari Dataset
# ============================
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)

    confidences = []
    for i in range(len(X_scaled)):
        true_label = y.iloc[i]
        pred_label = preds[i]
        prob = probs[i][np.argmax(probs[i])]
        if pred_label == true_label:
            confidences.append(prob)

    if confidences:
        avg_conf = np.mean(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)

        print("\n📊 Evaluasi Confidence di Dataset:")
        print(f" - Rata-rata confidence benar : {avg_conf:.2f}")
        print(f" - Confidence minimum benar   : {min_conf:.2f}")
        print(f" - Confidence maksimum benar  : {max_conf:.2f}")
    else:
        print("\n⚠️ Tidak ada prediksi yang benar di dataset untuk menghitung confidence.")
else:
    print("\n⚠️ Model tidak mendukung prediksi probabilitas (predict_proba).")
