import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--silver_features", type=str)
parser.add_argument("--train_output", type=str)
parser.add_argument("--test_output", type=str)
args = parser.parse_args()

print("Loading silver parquet:", args.silver_features)
df = pd.read_parquet(args.silver_features)

X = df.drop(columns=["label", "image_id"])
y = df["label"]

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=y, random_state=42
)

train_df.to_parquet(args.train_output, index=False)
test_df.to_parquet(args.test_output, index=False)

print("Train/Test split complete:")
print("Train:", train_df.shape)
print("Test:", test_df.shape)
