import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from predict import predict_with_baseline, MODEL_DIR

st.set_page_config(page_title="NLP Sentiment Classifier", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Classifier")
st.caption(
    "Classical TF-IDF baseline vs. a fine-tuned DistilBERT model — "
    "same transfer-learning idea as the plant disease project, applied to text."
)

review_text = st.text_area(
    "Paste a movie review:",
    height=150,
    placeholder="e.g. This film had incredible pacing and the performances were unforgettable...",
)

transformer_available = os.path.exists(MODEL_DIR)

if not transformer_available:
    st.warning(
        "No fine-tuned DistilBERT model found in `models/distilbert-sentiment`. "
        "Run `python src/train.py` first. Showing baseline-only predictions for now."
    )

if st.button("Classify", type="primary") and review_text.strip():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline (TF-IDF + LogReg)")
        try:
            label, confidence = predict_with_baseline(review_text)
            emoji = "😊" if label == "positive" else "😞"
            st.metric("Prediction", f"{emoji} {label}", f"{confidence:.1%} confidence")
        except FileNotFoundError:
            st.error("Baseline model not found. Run `python src/baseline.py` first.")

    with col2:
        st.subheader("Fine-tuned DistilBERT")
        if transformer_available:
            from predict import predict_with_transformer
            label, confidence = predict_with_transformer(review_text)
            emoji = "😊" if label == "positive" else "😞"
            st.metric("Prediction", f"{emoji} {label}", f"{confidence:.1%} confidence")
        else:
            st.info("Not trained yet.")

    if transformer_available:
        st.subheader("What the model focused on")
        try:
            from explain import get_word_importance, render_html_highlight

            pairs = get_word_importance(review_text)
            st.markdown(render_html_highlight(pairs), unsafe_allow_html=True)
            st.caption(
                "Darker highlight = the model's [CLS] token attended to that word more "
                "heavily when forming its final prediction (last-layer attention, averaged across heads)."
            )
        except Exception as e:
            st.caption(f"Word-importance view unavailable: {e}")

st.divider()
with st.expander("About this project"):
    st.markdown(
        """
        **Dataset:** IMDB Movie Reviews (50,000 labeled reviews)

        **Baseline:** TF-IDF vectorization + Logistic Regression

        **Main model:** `distilbert-base-uncased`, fine-tuned for binary
        sentiment classification

        **Why a baseline first?** It's easy to fine-tune a transformer and
        report a number. It's more useful to *prove* the transformer earns
        its extra training cost by comparing it against a fast, cheap
        classical model on the same test set. See `results/comparison.csv`
        for the full metrics table.
        """
    )
