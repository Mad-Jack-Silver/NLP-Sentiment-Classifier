import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baseline import BaselineSentimentModel, evaluate, train_and_evaluate_baseline

TRAIN_TEXTS = [
    "This movie was absolutely wonderful, I loved every minute of it",
    "Best film I have seen all year, brilliant acting",
    "A masterpiece, truly moving and well directed",
    "Terrible movie, a complete waste of time",
    "I hated this film, boring and way too long",
    "Awful plot and bad acting throughout",
] * 5
TRAIN_LABELS = [1, 1, 1, 0, 0, 0] * 5

TEST_TEXTS = [
    "Amazing movie, wonderful acting and great story",
    "Horrible film, boring and poorly written",
]
TEST_LABELS = [1, 0]


@pytest.fixture
def toy_dataframes():
    train_df = pd.DataFrame({"text": TRAIN_TEXTS, "label": TRAIN_LABELS})
    test_df = pd.DataFrame({"text": TEST_TEXTS, "label": TEST_LABELS})
    return train_df, test_df


def test_model_fits_and_predicts_correct_shape(toy_dataframes):
    train_df, test_df = toy_dataframes
    model = BaselineSentimentModel()
    model.fit(train_df["text"], train_df["label"])

    preds = model.predict(test_df["text"])
    assert len(preds) == len(test_df)
    assert set(preds).issubset({0, 1})


def test_model_predict_proba_sums_to_one(toy_dataframes):
    train_df, test_df = toy_dataframes
    model = BaselineSentimentModel()
    model.fit(train_df["text"], train_df["label"])

    probs = model.predict_proba(test_df["text"])
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-6


def test_evaluate_returns_expected_keys():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    metrics = evaluate(y_true, y_pred)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_and_evaluate_baseline_learns_the_obvious_pattern(toy_dataframes, tmp_path):
    train_df, test_df = toy_dataframes
    metrics = train_and_evaluate_baseline(train_df, test_df, save_model=False)
    assert metrics["accuracy"] >= 0.5


def test_save_and_load_roundtrip(toy_dataframes, tmp_path):
    train_df, _ = toy_dataframes
    model = BaselineSentimentModel()
    model.fit(train_df["text"], train_df["label"])

    save_path = tmp_path / "model.joblib"
    model.save(str(save_path))
    assert save_path.exists()

    loaded = BaselineSentimentModel.load(str(save_path))
    original_preds = model.predict(["great movie"])
    loaded_preds = loaded.predict(["great movie"])
    assert list(original_preds) == list(loaded_preds)
