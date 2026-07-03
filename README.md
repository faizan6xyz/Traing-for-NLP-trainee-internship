# Public Sentiment Analysis Pipeline

End-to-end sentiment analysis: text cleaning → TF-IDF / Word2Vec features →
Naive Bayes / Logistic Regression / LSTM → Streamlit dashboard showing
% Positive / Negative / Neutral.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Cleaning, tokenization, stopword removal (no NLTK download needed) |
| `features.py` | TF-IDF vectorizer + Word2Vec embeddings |
| `train.py` | Trains Naive Bayes, Logistic Regression, and an LSTM; saves everything to `./artifacts/` |
| `dashboard.py` | Streamlit app that loads the trained artifacts and predicts sentiment |
| `make_sample_data.py` | Generates a synthetic labeled dataset (`sample_data.csv`) so you can run the whole thing immediately |
| `sample_data.csv` | The generated demo dataset (750 rows, 3 balanced classes) |

## Quickstart

```bash
pip install -r requirements.txt

# 1. (optional) regenerate the demo dataset
python make_sample_data.py

# 2. train all three models on it
python train.py

# 3. launch the dashboard
streamlit run dashboard.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). You can type
a single sentence, or upload a CSV with a `text` column to get a batch
Positive/Negative/Neutral breakdown with a bar chart and a downloadable
results file.

## Using a real Kaggle dataset instead of the demo data

The demo dataset is synthetic (template-generated) so the pipeline runs
without any internet access to Kaggle. To use a real dataset, download one
of these from Kaggle and point `train.py` at it:

- **Twitter Sentiment140** — rename columns so you end up with `text` and
  `label` (label values 0/4 map to negative/positive; there's no neutral
  class in this dataset, so either drop the neutral logic in `dashboard.py`
  or map a middle score to neutral if you're using a different sentiment
  score column).
- **IMDB Reviews** — binary positive/negative. Columns are usually `review`
  and `sentiment`; rename `review` → `text`.
- **Amazon Product Reviews** — usually has a star rating; map 4–5★ →
  positive, 3★ → neutral, 1–2★ → negative, and rename the review text
  column to `text`.

Minimal required schema for `train.py`:

```csv
text,label
"This product is amazing!",positive
"Terrible, broke immediately.",negative
"It's fine, does the job.",neutral
```

Then run:

```bash
python train.py --data path/to/your_kaggle_file.csv
```

`train.py` auto-detects however many classes are in your `label` column
(2 for IMDB-style binary data, 3 for pos/neg/neutral), so no code changes
are needed for binary datasets — the dashboard's charts will just show
whichever classes exist.

## Notes on the models

- **Naive Bayes / Logistic Regression** run on TF-IDF (unigrams + bigrams,
  top 5000 features). These are fast, strong baselines for text
  classification and train in seconds even on CPU.
- **LSTM** uses a Word2Vec-initialized (and fine-tuned) embedding layer, a
  bidirectional single-layer LSTM, dropout, and a linear classifier head.
  It's implemented in PyTorch so it can use your RTX 2050 automatically —
  `train.py` calls `torch.cuda.is_available()` and moves the model there if
  present, no code changes needed. On a dataset the size of a full Kaggle
  Twitter Sentiment140 dump (1.6M rows), reduce to a random subsample
  (e.g. 100k–200k rows) for reasonable training time on 4GB VRAM, and keep
  `batch_size` modest (32–64).
- Accuracy on the bundled demo data is close to 100% because the synthetic
  templates are cleanly separable — that's expected and not representative
  of real-world text. Real Kaggle data will be noisier; realistic targets
  are roughly 75–85% for the classical models and comparable-to-slightly-better
  for the LSTM once you tune epochs/hidden size.

## Extending this

- Swap Word2Vec for pretrained GloVe/FastText vectors for a stronger LSTM
  starting point.
- Add a transformer baseline (e.g. `distilbert-base-uncased` via
  `transformers`) for comparison — you already have QLoRA fine-tuning
  experience, so this would be a natural next step for the portfolio.
- Add confusion matrix + per-class F1 plots to the dashboard sidebar (the
  `classification_report` output from `train.py` has everything needed).
