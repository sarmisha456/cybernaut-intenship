"""Load and clean the SMS Spam Collection csv (Kaggle format: v1=label, v2=text)."""
import pandas as pd

from .config import DATA_PATH


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]                      # drop junk 'Unnamed' columns
    df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df["target"] = (df["label"] == "spam").astype(int)   # spam=1, ham=0
    return df
