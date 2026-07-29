import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Removes HTML tags and extra whitespace from text."""
    
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_raw_dataset():
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    train_df = pd.DataFrame({
        "text": [clean_text(t) for t in ds["train"]["text"]],
        "label": ds["train"]["label"],
    })
    test_df = pd.DataFrame({
        "text": [clean_text(t) for t in ds["test"]["text"]],
        "label": ds["test"]["label"],
    })
    return train_df, test_df


def get_dataset(force_download: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_download and os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        return train_df, test_df

    train_df, test_df = load_raw_dataset()
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = get_dataset()
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(train_df["label"].value_counts())
    print(train_df.head(2))
