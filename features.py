
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from preprocess import preprocess_to_string, preprocess_text
def build_tfidf(train_texts, max_features=5000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(train_texts)
    return vectorizer, X
def build_word2vec(tokenized_texts, vector_size=100, window=5, min_count=2, epochs=20):
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,          # skip-gram (better for small datasets than CBOW)
        epochs=epochs,
        seed=42,
    )
    return model
def document_vector(tokens, w2v_model):
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)
def build_embedding_matrix(word2idx, w2v_model, vector_size=100):
    vocab_size = len(word2idx) + 1  # +1 for padding index 0
    matrix = np.zeros((vocab_size, vector_size), dtype=np.float32)
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
        else:
            matrix[idx] = np.random.normal(scale=0.1, size=(vector_size,))
    return matrix
if __name__ == "__main__":
    sample_texts = [
        "I loved this product, amazing quality",
        "Terrible experience, broke after one day",
        "It was okay, nothing special",
    ]
    cleaned = [preprocess_to_string(t) for t in sample_texts]
    vec, X = build_tfidf(cleaned)
    print("TF-IDF shape:", X.shape)
    print("Sample vocab:", list(vec.vocabulary_.items())[:5])
    tokenized = [preprocess_text(t) for t in sample_texts]
    w2v = build_word2vec(tokenized, vector_size=20, min_count=1, epochs=50)
    print("Word2Vec vocab size:", len(w2v.wv))
    print("Doc vector shape:", document_vector(tokenized[0], w2v).shape)
