# Movie Review Sentiment Classifier — Classical Baseline vs. Fine-Tuned DistilBERT

A binary sentiment classifier for movie reviews (IMDB dataset), built to answer
a specific question: **does fine-tuning a transformer actually earn its extra
training cost over a classical model, and by how much?**

This is my first NLP project, following the same transfer-learning idea used
in my [plant disease classifier](../plant-disease-classifier) — instead of
training a model from scratch, a model already pretrained on a huge corpus
(`distilbert-base-uncased`) is fine-tuned on this specific task. Same
principle, applied to text instead of images.

## Why a baseline first?

It's easy to fine-tune a transformer and report one accuracy number. It's
more useful to prove the transformer is worth using — this project trains a
classical **TF-IDF + Logistic Regression** model first, evaluates it on the
same held-out test set, and only then compares it against the fine-tuned
transformer. The comparison table below is the actual point of the project.

## Results

Evaluated on the 25k-review IMDB held-out test set (`src/evaluate.py`,
saved to `results/comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8819 | 0.8792 | 0.8854 | 0.8823 |
| Fine-tuned DistilBERT | 0.8506 | 0.8573 | 0.8411 | 0.8491 |

**So does the transformer earn its extra training cost here? No — not in
this run.** The classical TF-IDF + Logistic Regression baseline beats the
fine-tuned DistilBERT on all four metrics (roughly 2-4 points higher across
the board), despite being far cheaper to train. If you re-run `src/train.py`
as-is, note that its `__main__` block trains on a 500-example subsample for
1 epoch (the "quick smoke test" mode mentioned above) rather than the full
25k train set for 2 epochs described in Method details — that's the most
likely reason the transformer underperforms here. Training on the full
dataset (`train_distilbert(train_df, test_df, epochs=2, subsample=None)`)
would be the natural next step to see whether DistilBERT actually pulls
ahead once it's given a fair amount of data.

## Project structure

```
nlp-sentiment-classifier/
├── app.py                    # Streamlit demo
├── requirements.txt
├── data/                     # IMDB dataset cache (downloaded on first run)
├── models/                   # saved baseline + fine-tuned model
├── results/                  # comparison.csv, saved after evaluate.py
├── src/
│   ├── data_loader.py        # download, clean, cache the IMDB dataset
│   ├── baseline.py           # TF-IDF + Logistic Regression
│   ├── train.py               # DistilBERT fine-tuning
│   ├── evaluate.py           # baseline vs transformer comparison
│   ├── predict.py            # CLI inference (either model)
│   └── explain.py            # attention-based word importance (Grad-CAM equivalent for text)
└── tests/                    # unit tests (run offline, no download needed)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** `src/train.py` needs a GPU with a few GB of VRAM for a reasonable
training time (CPU works but is slow — a subsample option is included for a
quick smoke test). It also needs an internet connection on first run to
download the pretrained `distilbert-base-uncased` weights (~260MB) and the
IMDB dataset (~80MB), same as the plant disease project's GPU note.

## How to run, step by step

```bash
# 1. Download + cache the dataset
python src/data_loader.py

# 2. Train and evaluate the classical baseline (fast, runs on CPU in seconds)
python src/baseline.py

# 3. Fine-tune DistilBERT (needs GPU for a reasonable runtime)
python src/train.py

# 4. Generate the baseline vs transformer comparison table
python src/evaluate.py

# 5. Try a single prediction from the command line
python src/predict.py "This movie completely wasted my time."

# 6. Launch the interactive demo
streamlit run app.py
```

## Method details

- **Dataset:** IMDB Movie Reviews — 50,000 labeled reviews (25k train / 25k
  test), balanced between positive and negative.
- **Preprocessing:** HTML artifact stripping (`<br />` tags) and whitespace
  normalization only — no lowercasing/stopword removal for the transformer
  input, since DistilBERT's tokenizer handles that internally. The TF-IDF
  baseline applies its own lowercasing and stopword removal separately.
- **Baseline:** TF-IDF (unigrams + bigrams, 10k features) + Logistic
  Regression.
- **Main model:** `distilbert-base-uncased` fine-tuned for 2 epochs with the
  HuggingFace `Trainer` API, max sequence length 256.
- **Interpretability:** last-layer attention from the `[CLS]` token to each
  input token, averaged across attention heads, rendered as a highlight
  intensity per word in the Streamlit demo.

## Tests

```bash
pytest tests/ -v
```

Tests run entirely offline against small synthetic examples — they check the
pipeline mechanics (shapes, save/load roundtrips, metric keys, CLI fallback
behavior), not model quality on the real dataset.