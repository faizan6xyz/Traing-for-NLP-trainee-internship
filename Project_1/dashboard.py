import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from preprocess import preprocess_text, preprocess_to_string
from train import SentimentLSTM, encode_and_pad
ARTIFACT_DIR = "Project_1/artifacts"
@st.cache_resource
def load_artifacts():
    with open(f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/naive_bayes.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/logistic_regression.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(f"{ARTIFACT_DIR}/word2idx.json", "r") as f:
        word2idx = json.load(f)
    with open(f"{ARTIFACT_DIR}/summary.json", "r") as f:
        summary = json.load(f)
    checkpoint = torch.load(f"{ARTIFACT_DIR}/lstm_model.pt", map_location="cpu")
    vocab_size, embed_dim = checkpoint["embedding_matrix_shape"]
    dummy_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    lstm_model = SentimentLSTM(dummy_matrix, num_classes=checkpoint["num_classes"])
    lstm_model.load_state_dict(checkpoint["model_state"])
    lstm_model.eval()
    return {
        "vectorizer": vectorizer,
        "nb_model": nb_model,
        "lr_model": lr_model,
        "label_encoder": label_encoder,
        "word2idx": word2idx,
        "lstm_model": lstm_model,
        "max_len": checkpoint["max_len"],
        "summary": summary,
    }
def predict_batch(texts, artifacts, model_choice):
    le = artifacts["label_encoder"]

    if model_choice in ("Naive Bayes", "Logistic Regression"):
        cleaned = [preprocess_to_string(t) for t in texts]
        X = artifacts["vectorizer"].transform(cleaned)
        model = artifacts["nb_model"] if model_choice == "Naive Bayes" else artifacts["lr_model"]
        preds = model.predict(X)
        labels = le.inverse_transform(preds)
        return list(labels)
    #lstm
    tokenized = [preprocess_text(t) for t in texts]
    X, lengths = encode_and_pad(tokenized, artifacts["word2idx"], artifacts["max_len"])
    with torch.no_grad():
        logits = artifacts["lstm_model"](X, lengths)
        preds = logits.argmax(dim=1).numpy()
    labels = le.inverse_transform(preds)
    return list(labels)
def render_distribution(labels):
    counts = pd.Series(labels).value_counts(normalize=True) * 100
    counts = counts.reindex(["positive", "neutral", "negative"]).fillna(0)

    col1, col2, col3 = st.columns(3)
    col1.metric(" Positive", f"{counts.get('positive', 0):.1f}%")
    col2.metric(" Neutral", f"{counts.get('neutral', 0):.1f}%")
    col3.metric(" Negative", f"{counts.get('negative', 0):.1f}%")

    chart_df = pd.DataFrame({"sentiment": counts.index, "percentage": counts.values})
    st.bar_chart(chart_df.set_index("sentiment"))
def main():
    st.set_page_config(page_title="Sentiment Dashboard", page_icon="💬", layout="centered")
    st.title(" Public Sentiment Dashboard")
    st.caption("TF-IDF + Naive Bayes / Logistic Regression, and Word2Vec + LSTM")
    try:
        artifacts = load_artifacts()
    except FileNotFoundError:
        st.error(
            "No trained artifacts found. Run `python train.py` first "
            "(it will create the ./artifacts folder used by this dashboard)."
        )
        st.stop()
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox(
            "Model", ["Logistic Regression", "Naive Bayes", "LSTM"], index=0
        )
        st.divider()
        st.subheader("Last training run")
        st.json(artifacts["summary"])

    tab1, tab2 = st.tabs([" Single text", " Batch CSV"])
    with tab1:
        text_input = st.text_area(
            "Enter a tweet / review to analyze",
            placeholder="e.g. This product completely exceeded my expectations!",
        )
        if st.button("Analyze", type="primary", key="single"):
            if not text_input.strip():
                st.warning("Please enter some text.")
            else:
                label = predict_batch([text_input], artifacts, model_choice)[0]
                emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(label, "")
                st.success(f"Predicted sentiment: **{label.upper()}** {emoji}")
    with tab2:
        uploaded = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                with st.spinner("Analyzing..."):
                    df["predicted_sentiment"] = predict_batch(
                        df["text"].astype(str).tolist(), artifacts, model_choice
                    )
                st.subheader("Overall sentiment breakdown")
                render_distribution(df["predicted_sentiment"].tolist())
                st.subheader("Predictions")
                st.dataframe(df[["text", "predicted_sentiment"]], use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "sentiment_results.csv",
                    "text/csv",
                )
if __name__ == "__main__":
    main()
# run it in the the termianl with "streamlit run project_1/dashboard.py"