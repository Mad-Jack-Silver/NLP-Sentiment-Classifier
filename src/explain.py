import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert-sentiment")


def get_word_importance(text: str):
    """Returns a list of (token, importance_score) tuples, scores normalized 0-1."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, output_attentions=True)
    model.to(device)
    model.eval()

    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    last_layer_attn = outputs.attentions[-1][0].cpu()          # (num_heads, seq_len, seq_len)
    cls_attn = last_layer_attn[:, 0, :].mean(dim=0)             # avg across heads, CLS row -> (seq_len,)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    scores = cls_attn.numpy()

    # Drop [CLS] and [SEP] from the display, they're structural not content words
    pairs = [(t, s) for t, s in zip(tokens, scores) if t not in ("[CLS]", "[SEP]", "[PAD]")]
    if pairs:
        max_score = max(s for _, s in pairs) or 1.0
        pairs = [(t, s / max_score) for t, s in pairs]

    return pairs


def render_html_highlight(pairs) -> str:
    spans = []
    for token, score in pairs:
        display_token = token.replace("##", "")
        alpha = round(float(score), 2)
        spans.append(
            f'<span style="background-color: rgba(255, 99, 71, {alpha}); '
            f'padding: 2px 3px; border-radius: 3px; margin: 1px;">{display_token}</span>'
        )
    return " ".join(spans)


if __name__ == "__main__":
    sample = "This movie was surprisingly boring and way too long for what it delivered."
    pairs = get_word_importance(sample)
    for token, score in pairs:
        print(f"{token:>15s}  {score:.3f}")
