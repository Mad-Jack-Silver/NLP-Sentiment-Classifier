import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import baseline
import predict


def test_predict_falls_back_to_baseline_when_transformer_missing(tmp_path, monkeypatch):
    fake_baseline_path = tmp_path / "baseline.joblib"
    fake_transformer_dir = tmp_path / "distilbert-sentiment"  # deliberately never created

    monkeypatch.setattr(baseline, "BASELINE_PATH", str(fake_baseline_path))
    monkeypatch.setattr(predict, "MODEL_DIR", str(fake_transformer_dir))

    train_texts = [
        "wonderful and brilliant film", "great acting, loved it",
        "terrible and boring movie", "awful, a complete waste of time",
    ] * 5
    train_labels = [1, 1, 0, 0] * 5

    model = baseline.BaselineSentimentModel()
    model.fit(train_texts, train_labels)
    model.save(str(fake_baseline_path))

    label, confidence = predict.predict("wonderful and brilliant", use_model="transformer")

    assert label in ("positive", "negative")
    assert 0.0 <= confidence <= 1.0


def test_predict_explicit_baseline_mode(tmp_path, monkeypatch):
    fake_baseline_path = tmp_path / "baseline.joblib"
    monkeypatch.setattr(baseline, "BASELINE_PATH", str(fake_baseline_path))

    train_texts = ["amazing and wonderful", "horrible and bad"] * 5
    train_labels = [1, 0] * 5

    model = baseline.BaselineSentimentModel()
    model.fit(train_texts, train_labels)
    model.save(str(fake_baseline_path))

    label, confidence = predict.predict("amazing", use_model="baseline")
    assert label in ("positive", "negative")
