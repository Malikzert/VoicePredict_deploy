import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("data/features_buka_tutup.csv")

X = df.drop(columns=["file_name", "class"])
y = df["class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Akurasi:", model.score(X_test, y_test))

# Simpan model
joblib.dump(model, "models/classifier.pkl")
print("Model disimpan ke models/classifier.pkl (berhasil disimpan)")


