import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
COMPARISON_CSV = os.path.join(RESULTS_DIR, "comparison.csv")


def evaluate_baseline_on_test(test_df):
    from baseline import BaselineSentimentModel

    model = BaselineSentimentModel.load()
    preds = model.predict(test_df["text"])
    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy_score(test_df["label"], preds),
        "precision": precision_score(test_df["label"], preds),
        "recall": recall_score(test_df["label"], preds),
        "f1": f1_score(test_df["label"], preds),
    }


def evaluate_distilbert_on_test(test_df):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DistilBERT evaluation on: {device}")

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert-sentiment")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    preds = []
    batch_size = 32
    texts = test_df["text"].tolist()
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Evaluating DistilBERT"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "model": "Fine-tuned DistilBERT",
        "accuracy": accuracy_score(test_df["label"], preds),
        "precision": precision_score(test_df["label"], preds),
        "recall": recall_score(test_df["label"], preds),
        "f1": f1_score(test_df["label"], preds),
    }


def build_comparison_table(baseline_metrics: dict, transformer_metrics: dict) -> pd.DataFrame:
    df = pd.DataFrame([baseline_metrics, transformer_metrics])
    df = df.set_index("model")
    df["improvement_over_baseline"] = ""
    for col in ["accuracy", "precision", "recall", "f1"]:
        delta = df.loc["Fine-tuned DistilBERT", col] - df.loc["TF-IDF + Logistic Regression", col]
        df.loc["Fine-tuned DistilBERT", "improvement_over_baseline"] = (
            df.loc["Fine-tuned DistilBERT", "improvement_over_baseline"]
        )
    return df


if __name__ == "__main__":
    from data_loader import get_dataset

    _, test_df = get_dataset()

    baseline_metrics = evaluate_baseline_on_test(test_df)
    transformer_metrics = evaluate_distilbert_on_test(test_df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison = pd.DataFrame([baseline_metrics, transformer_metrics])
    comparison.to_csv(COMPARISON_CSV, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved comparison table to {COMPARISON_CSV}")
