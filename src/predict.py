import os
import argparse

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert-sentiment")

LABELS = {0: "negative", 1: "positive"}


def predict_with_transformer(text: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    pred_label = int(logits.argmax(dim=1).item())
    return LABELS[pred_label], probs[pred_label]


def predict_with_baseline(text: str):
    from baseline import BaselineSentimentModel

    model = BaselineSentimentModel.load()
    pred_label = int(model.predict([text])[0])
    proba = model.predict_proba([text])[0]
    confidence = proba[pred_label]
    return LABELS[pred_label], confidence


def predict(text: str, use_model: str = "transformer"):
    if use_model == "transformer" and os.path.exists(MODEL_DIR):
        return predict_with_transformer(text)
    return predict_with_baseline(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment of a movie review.")
    parser.add_argument("text", type=str, help="Review text to classify")
    parser.add_argument("--model", choices=["transformer", "baseline"], default="transformer")
    args = parser.parse_args()

    label, confidence = predict(args.text, use_model=args.model)
    print(f"Prediction: {label}  (confidence: {confidence:.2%})")
