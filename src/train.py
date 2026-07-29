import os
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(MODELS_DIR, "distilbert-sentiment")
MAX_LENGTH = 256


class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def train_distilbert(train_df, test_df, epochs: int = 2, batch_size: int = 16,
                      subsample: int = None):
    if subsample:
        train_df = train_df.sample(n=subsample, random_state=42).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = ReviewsDataset(train_df["text"], train_df["label"], tokenizer)
    test_dataset = ReviewsDataset(test_df["text"], test_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return metrics


if __name__ == "__main__":
    from data_loader import get_dataset

    train_df, test_df = get_dataset()
    metrics = train_distilbert(train_df, test_df, epochs=1, batch_size=16, subsample=500)
    print("DistilBERT fine-tuned results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")