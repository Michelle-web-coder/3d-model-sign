import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Path to collected data
DATA_FOLDER = "collected_data"
X = []
y = []

expected_length = None  # Set this from the first valid row

print("📥 Loading data from:", DATA_FOLDER)

# Load CSV files
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        file_path = os.path.join(DATA_FOLDER, file)
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    # Skip header
                    print(f"⚠️ Skipping header row in {file}")
                    continue
                if len(row) < 2:
                    continue  # Skip empty/invalid rows
                try:
                    features = [float(val) for val in row[1:]]
                    # Set expected length on first valid row
                    if expected_length is None:
                        expected_length = len(features)
                    # Ensure consistent shape
                    if len(features) != expected_length:
                        print(f"⚠️ Skipping row with inconsistent length in {file}: {len(features)} features")
                        continue
                    y.append(row[0])
                    X.append(features)
                except ValueError:
                    print(f"⚠️ Skipping non-numeric row in {file}: {row}")

# Convert to arrays
X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples across {len(set(y))} labels.")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("💾 Model and label map saved successfully!")
