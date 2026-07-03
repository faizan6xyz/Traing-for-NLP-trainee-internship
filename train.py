import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_text, preprocess_to_string
from features import build_tfidf, build_word2vec, build_embedding_matrix
ARTIFACT_DIR = "artifacts"
class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=64, num_classes=3, pad_idx=0):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=False, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.fc(self.dropout(h_cat))
        return out
def encode_and_pad(tokenized_texts, word2idx, max_len=40):
    seqs = []
    lengths = []
    for tokens in tokenized_texts:
        ids = [word2idx.get(t, 0) for t in tokens][:max_len]
        length = max(len(ids), 1)
        ids = ids + [0] * (max_len - len(ids))
        seqs.append(ids)
        lengths.append(length)
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
def train_lstm(X_train_tok, y_train, X_test_tok, y_test, word2idx, embedding_matrix,
                num_classes, epochs=12, batch_size=32, lr=1e-3, max_len=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, len_train = encode_and_pad(X_train_tok, word2idx, max_len)
    X_test, len_test = encode_and_pad(X_test_tok, word2idx, max_len)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    model = SentimentLSTM(embedding_matrix, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = X_train.size(0)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, lb, yb = X_train[idx].to(device), len_train[idx], y_train_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / n
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device), len_test)
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        print(f"  epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}")
    return model
def main(data_path: str):
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")
    print("\nPreprocessing text ...")
    df["clean_str"] = df["text"].apply(preprocess_to_string)
    df["clean_tokens"] = df["text"].apply(preprocess_text)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    num_classes = len(le.classes_)
    print("Classes:", list(le.classes_))
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    print("\n[1/3] Training Naive Bayes + Logistic Regression on TF-IDF features ...")
    vectorizer, X_train_tfidf = build_tfidf(train_df["clean_str"].tolist())
    X_test_tfidf = vectorizer.transform(test_df["clean_str"].tolist())
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, train_df["label_id"])
    nb_preds = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(test_df["label_id"], nb_preds)
    print(f"  Naive Bayes accuracy: {nb_acc:.4f}")
    print(classification_report(test_df["label_id"], nb_preds, target_names=le.classes_))
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, train_df["label_id"])
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(test_df["label_id"], lr_preds)
    print(f"  Logistic Regression accuracy: {lr_acc:.4f}")
    print(classification_report(test_df["label_id"], lr_preds, target_names=le.classes_))
    print("[2/3] Training Word2Vec embeddings ...")
    w2v = build_word2vec(df["clean_tokens"].tolist(), vector_size=100, min_count=1)
    vocab = sorted(w2v.wv.index_to_key)
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}  # 0 reserved for padding
    embedding_matrix = build_embedding_matrix(word2idx, w2v, vector_size=100)
    print(f"  Word2Vec vocab size: {len(vocab)}")
    print("\n[3/3] Training LSTM ...")
    lstm_model = train_lstm(
        train_df["clean_tokens"].tolist(), train_df["label_id"].tolist(),
        test_df["clean_tokens"].tolist(), test_df["label_id"].tolist(),
        word2idx, embedding_matrix, num_classes=num_classes,
        epochs=12,
    )
    import os
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{ARTIFACT_DIR}/naive_bayes.pkl", "wb") as f:
        pickle.dump(nb, f)
    with open(f"{ARTIFACT_DIR}/logistic_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(f"{ARTIFACT_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(f"{ARTIFACT_DIR}/word2idx.json", "w") as f:
        json.dump(word2idx, f)
    torch.save(
        {
            "model_state": lstm_model.state_dict(),
            "embedding_matrix_shape": embedding_matrix.shape,
            "num_classes": num_classes,
            "max_len": 40,
        },
        f"{ARTIFACT_DIR}/lstm_model.pt",
    )

    summary = {
        "naive_bayes_accuracy": round(nb_acc, 4),
        "logistic_regression_accuracy": round(lr_acc, 4),
        "classes": list(le.classes_),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    with open(f"{ARTIFACT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll artifacts saved to ./artifacts/")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="twitter_training.csv",
                         help="Path to CSV with 'text' and 'label' columns")
    args = parser.parse_args()
    main(args.data)
