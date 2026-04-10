from pathlib import Path
import pandas as pd
import yaml
from scipy.io import arff
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/dataset")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

split_params = params["split"]

data, meta = arff.loadarff(RAW_PATH)
df = pd.DataFrame(data)

# если target в bytes
if df["Class"].dtype == "object":
    df["Class"] = df["Class"].apply(
        lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x)
    )

train_df, test_df = train_test_split(
    df,
    test_size=split_params["test_size"],
    random_state=split_params["random_state"],
    stratify=df["Class"] if split_params.get("stratify", False) else None,
)

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)