# Trainee Internship Projects

Three self-contained machine learning / deep learning projects completed as part of an NLP trainee internship. Each project lives in its own folder with its own scripts, notebook, and (where applicable) saved model artifacts.

| # | Project | Task | Core techniques |
|---|---------|------|------------------|
| 1 | [`Project_1`](#project-1--twitter-sentiment-analysis) | Twitter/text sentiment classification | Text preprocessing, TF-IDF, Word2Vec, Naive Bayes, Logistic Regression, LSTM, Streamlit dashboard |
| 2 | [`project_2`](#project-2--customer-segmentation--recommendations) | Customer segmentation & recommendations | K-Means, DBSCAN, silhouette scoring, cosine-similarity recommenders, 2D/3D visualization |
| 3 | [`project_3`](#project-3--cat-vs-dog-image-classifier) | Cat vs. Dog image classification | CNN (TensorFlow/Keras), data augmentation, corrupt-image filtering |

---

## Project 1 — Twitter Sentiment Analysis

**Folder:** `Project_1/`

An end-to-end sentiment classification pipeline that trains and compares three models on the same cleaned text data, plus an interactive Streamlit dashboard to try them out.

**Pipeline**
- `preprocess.py` — lowercases text, strips HTML/URLs/@mentions, unwraps hashtags, removes punctuation and stopwords, and applies a lightweight custom suffix stemmer.
- `features.py` — builds TF-IDF vectors (unigrams + bigrams) and a skip-gram Word2Vec model, plus helpers to turn tokens into an embedding matrix.
- `train.py` — trains and evaluates three models on the same train/test split:
  - Multinomial Naive Bayes (TF-IDF)
  - Logistic Regression (TF-IDF)
  - A bidirectional LSTM (PyTorch) on Word2Vec embeddings
  
  Saves all trained artifacts (vectorizer, models, label encoder, vocab, LSTM weights, a `summary.json` of accuracies) to `Project_1/artifacts/`.
- `dashboard.py` — a Streamlit app that loads the saved artifacts, lets you pick a model, paste in text, and view predicted sentiment plus a distribution chart.
- `make_sample_data.py` — generates a synthetic labeled dataset (`positive` / `negative` / `neutral`) from templated sentences, useful for a quick end-to-end test run without a real dataset.
- `training.ipynb` — notebook walkthrough of the same pipeline.
- `twitter_training.csv` — the training dataset (`text`, `label` columns).

**Run it**
```bash
cd Project_1
python train.py --data twitter_training.csv   # trains all 3 models, saves artifacts/
streamlit run dashboard.py                    # launch the interactive dashboard
```

---

## Project 2 — Customer Segmentation & Recommendations

**Folder:** `project_2/`

Segments synthetic customer data (age, income, spending score) into clusters and builds two simple recommendation approaches on top of the segments.

**What it does**
- Generates a synthetic customer dataset (`load_data`) with `Age`, `Annual_Income`, `Spending_Score`, `Gender`.
- Scales features and finds the best K for **K-Means** via silhouette score sweep (k = 2–8).
- Estimates an epsilon for **DBSCAN** automatically using a k-distance "elbow" (knee-point) method, then clusters with it.
- Labels each customer into a human-readable segment (e.g. *High Income – High Spending*) based on median income/spending splits.
- Builds a synthetic purchase-interaction matrix and generates recommendations two ways:
  - **Item-based collaborative filtering** (cosine similarity between products).
  - **Segment/profile similarity** (cosine similarity between customers in the same segment).
- Produces 2D/3D visualizations (Matplotlib static + Plotly interactive) and CSV exports.

**Files**
- `customer_segmentation.py` — the full pipeline described above.
- `training.ipynb` — notebook version.
- `Mall_Customers.csv`, `new.csv`, `customer_segments.csv` — datasets/outputs.
- `output_plot/` — saved charts: K-Means/DBSCAN 2D & 3D scatter plots, elbow plot, silhouette scores.

**Run it**
```bash
cd project_2
python customer_segmentation.py
```
> Note: the script writes outputs to `/mnt/user-data/outputs` by default (`OUT` constant) — update this path if running outside that environment.

---

## Project 3 — Cat vs. Dog Image Classifier

**Folder:** `project_3/`

A convolutional neural network (TensorFlow/Keras) trained to classify images as **Cat** or **Dog**, based on the classic Kaggle "Dogs vs. Cats" (PetImages) dataset.

**What it does**
- `delete_corruptfile.py` — scans the dataset directory and detects/removes corrupt or truncated image files (a well-known issue with the PetImages dataset) before training.
- `Train.py`:
  - Loads images from `PetImages/Cat` and `PetImages/Dog` via `image_dataset_from_directory` (binary labels, auto 80/20 train/val split).
  - Applies data augmentation (random flip, rotation, zoom) to the training set only.
  - Trains a 4-block CNN (Conv2D + MaxPooling stack → dense head → sigmoid output) with early stopping and learning-rate reduction on plateau.
  - Saves the trained model to `cat_dog_model.h5` and the class name order to `class_names.txt`.
- `PetImages/` — the raw image dataset (`Cat/`, `Dog/` subfolders, ~12,000+ images each).

**Run it**
```bash
cd project_3
python delete_corruptfile.py   # optional cleanup pass first
python Train.py
```

---

## Requirements

Across the three projects, the main dependencies are:

```
numpy
pandas
scikit-learn
gensim
torch
tensorflow
streamlit
matplotlib
plotly
pillow
```

No `requirements.txt` is currently included — install the libraries relevant to whichever project you're running (e.g. `torch` + `gensim` for Project 1, `tensorflow` for Project 3).

## Repository structure

```
.
├── Project_1/          # Sentiment analysis (NB, Logistic Regression, LSTM) + Streamlit dashboard
├── project_2/           # Customer segmentation (K-Means/DBSCAN) + recommenders
└── project_3/            # Cat vs Dog CNN image classifier
```