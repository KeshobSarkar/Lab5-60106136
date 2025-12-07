import argparse
import os
import json
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import mlflow
import mlflow.sklearn

# ---------------- ARGPARSE ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_input")
parser.add_argument("--test_input")
parser.add_argument("--selected_features")
parser.add_argument("--metrics_output")
parser.add_argument("--model_output")  # MLflow output folder
args = parser.parse_args()

# ---------------- LOAD DATA ----------------
print("Loading training data from:", args.train_input)
train_df = pd.read_parquet(args.train_input)

print("Loading test data from:", args.test_input)
test_df = pd.read_parquet(args.test_input)

# ---------------- LOAD SELECTED FEATURES ----------------
print("Loading selected features JSON:", args.selected_features)
with open(args.selected_features, "r") as f:
    selected = json.load(f)["selected_features"]

print("Selected features:", selected)

# ---------------- TRAIN MODEL ----------------
X_train = train_df[selected]
y_train = train_df["label"]
X_test = test_df[selected]
y_test = test_df["label"]

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()

print("Accuracy:", acc)
print("Confusion Matrix:", cm)

# ---------------- SAVE METRICS.JSON ----------------
os.makedirs(args.metrics_output, exist_ok=True)
metrics_path = os.path.join(args.metrics_output, "metrics.json")

metrics = {
    "accuracy": float(acc),
    "confusion_matrix": cm,
    "num_features": len(selected),
    "selected_features": selected,
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print("Saved metrics to:", metrics_path)

# ---------------- SAVE MODEL (MLflow Format) ----------------
print("Saving MLflow model to:", args.model_output)

# Save model folder in MLflow format
mlflow.sklearn.save_model(clf, path=args.model_output)

# ---------------- ADD METADATA ----------------
metadata = {
    "selected_features": selected,
    "training_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "feature_set_version": "1",
}

metadata_path = os.path.join(args.model_output, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f)

print("Metadata saved:", metadata_path)
print("✅ Train/Eval complete + Model registered!")
