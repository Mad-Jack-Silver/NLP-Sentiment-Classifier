import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASELINE_PATH = os.path.join(MODELS_DIR, "baseline_tfidf_logreg.joblib")


class BaselineSentimentModel:
    def __init__(self, max_features: int = 10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self.clf = LogisticRegression(max_iter=1000, C=1.0)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)

    def save(self, path: str = None):
        path = path or BASELINE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str = None):
        path = path or BASELINE_PATH
        obj = cls()
        bundle = joblib.load(path)
        obj.vectorizer = bundle["vectorizer"]
        obj.clf = bundle["clf"]
        return obj


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_and_evaluate_baseline(train_df, test_df, save_model: bool = True) -> dict:
    model = BaselineSentimentModel()
    model.fit(train_df["text"], train_df["label"])

    preds = model.predict(test_df["text"])
    metrics = evaluate(test_df["label"], preds)

    if save_model:
        model.save()

    return metrics


if __name__ == "__main__":
    from data_loader import get_dataset

    train_df, test_df = get_dataset()
    metrics = train_and_evaluate_baseline(train_df, test_df)
    print("Baseline (TF-IDF + Logistic Regression) results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")