import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
RNG = np.random.default_rng(42)
OUT = "/mnt/user-data/outputs"
def load_data(n=400):
    age = RNG.integers(18, 70, n)
    income = np.clip(RNG.normal(60, 25, n), 15, 140)        
    spending = np.clip(
        50 - 0.3 * (income - 60) + RNG.normal(0, 25, n), 1, 100
    )
    gender = RNG.choice(["Male", "Female"], n)
    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Gender": gender,
        "Age": age,
        "Annual_Income": income.round(1),
        "Spending_Score": spending.round(1),
    })
    return df
def scale_features(df, cols=("Annual_Income", "Spending_Score", "Age")):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[list(cols)])
    return X, scaler
def best_kmeans_k(X, k_range=range(2, 9)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    best_k = max(scores, key=scores.get)
    return best_k, scores
def run_kmeans(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, km
def estimate_dbscan_eps(X, k=5):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, _ = nn.kneighbors(X)
    k_dist = np.sort(dist[:, -1])
    x = np.linspace(0, 1, len(k_dist))
    y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min() + 1e-9)
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec /= np.linalg.norm(line_vec)
    diffs = np.stack([x - x[0], y - y[0]], axis=1)
    proj_len = diffs @ line_vec
    proj = np.outer(proj_len, line_vec)
    dist_to_line = np.linalg.norm(diffs - proj, axis=1)
    knee_idx = np.argmax(dist_to_line)
    return k_dist[knee_idx], k_dist
def run_dbscan(X, eps, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db
def label_segments(df, income_col="Annual_Income", spend_col="Spending_Score"):
    inc_med = df[income_col].median()
    spend_med = df[spend_col].median()
    def tag(row):
        hi_income = row[income_col] >= inc_med
        hi_spend = row[spend_col] >= spend_med
        if hi_income and hi_spend:
            return "High Income - High Spending"
        if hi_income and not hi_spend:
            return "High Income - Low Spending"
        if not hi_income and hi_spend:
            return "Low Income - High Spending"
        return "Low Income - Low Spending"

    df["Segment"] = df.apply(tag, axis=1)
    return df
def build_synthetic_interactions(df, n_products=15):
    products = [f"Product_{i}" for i in range(1, n_products + 1)]
    n = len(df)
    mat = np.zeros((n, n_products))
    for i, row in df.iterrows():
        n_purchases = RNG.integers(2, 6)
        weights = np.linspace(1, row["Spending_Score"] / 20 + 1, n_products)
        weights /= weights.sum()
        chosen = RNG.choice(n_products, size=n_purchases, replace=False, p=weights)
        mat[i, chosen] = 1
    return pd.DataFrame(mat, index=df["CustomerID"], columns=products)
def item_based_recommend(interactions, customer_id, top_n=3):
    item_sim = cosine_similarity(interactions.T)
    item_sim_df = pd.DataFrame(item_sim, index=interactions.columns, columns=interactions.columns)
    user_row = interactions.loc[customer_id]
    purchased = user_row[user_row > 0].index.tolist()
    if not purchased:
        return []
    scores = item_sim_df[purchased].sum(axis=1).drop(labels=purchased, errors="ignore")
    return scores.sort_values(ascending=False).head(top_n).index.tolist()
def segment_similarity_recommend(df, customer_id, top_n=5):
    target = df[df["CustomerID"] == customer_id].iloc[0]
    same_seg = df[(df["Segment"] == target["Segment"]) & (df["CustomerID"] != customer_id)]
    feats = ["Age", "Annual_Income", "Spending_Score"]
    sims = cosine_similarity(
        StandardScaler().fit_transform(df[feats]),
    )
    sims_df = pd.DataFrame(sims, index=df["CustomerID"], columns=df["CustomerID"])
    ranked = sims_df.loc[customer_id, same_seg["CustomerID"]].sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()
def plot_2d_matplotlib(df, label_col, fname):
    plt.figure(figsize=(8, 6))
    labels = df[label_col].unique()
    for lab in sorted(labels, key=str):
        sub = df[df[label_col] == lab]
        plt.scatter(sub["Annual_Income"], sub["Spending_Score"], label=str(lab), s=40, alpha=0.75)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    plt.title(f"Customer Clusters ({label_col})")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
def plot_3d_matplotlib(df, label_col, fname):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    labels = sorted(df[label_col].unique(), key=str)
    for lab in labels:
        sub = df[df[label_col] == lab]
        ax.scatter(sub["Age"], sub["Annual_Income"], sub["Spending_Score"], label=str(lab), s=30, alpha=0.75)
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score")
    ax.set_title(f"3D Customer Clusters ({label_col})")
    ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
def plot_3d_plotly(df, label_col, fname):
    fig = px.scatter_3d(
        df, x="Age", y="Annual_Income", z="Spending_Score",
        color=df[label_col].astype(str),
        hover_data=["CustomerID", "Gender", "Segment"],
        title=f"Interactive 3D Customer Clusters ({label_col})",
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=5))
    fig.write_html(fname)
if __name__ == "__main__":
    df = load_data()
    # --- clustering ---
    X, scaler = scale_features(df)
    best_k, sil_scores = best_kmeans_k(X)
    print("Silhouette scores by k:", {k: round(v, 3) for k, v in sil_scores.items()})
    print("Chosen k for K-Means:", best_k)
    df["KMeans_Cluster"], kmeans_model = run_kmeans(X, best_k)
    eps, _ = estimate_dbscan_eps(X)
    print("Estimated DBSCAN eps:", round(eps, 3))
    df["DBSCAN_Cluster"], dbscan_model = run_dbscan(X, eps=eps, min_samples=5)
    n_noise = (df["DBSCAN_Cluster"] == -1).sum()
    print(f"DBSCAN found {df['DBSCAN_Cluster'].nunique() - (1 if n_noise else 0)} clusters, {n_noise} noise points")
    # --- segment labeling ---
    df = label_segments(df)
    print("\nSegment counts:\n", df["Segment"].value_counts())
    # --- recommendation engine ---
    interactions = build_synthetic_interactions(df)
    sample_id = df["CustomerID"].iloc[0]
    item_recs = item_based_recommend(interactions, sample_id)
    profile_recs = segment_similarity_recommend(df, sample_id)
    print(f"\nCustomer {sample_id} segment: {df.loc[df.CustomerID == sample_id, 'Segment'].values[0]}")
    print("Item-based CF recommendations:", item_recs)
    print("Similar customers in same segment:", profile_recs)
    plot_2d_matplotlib(df, "KMeans_Cluster", f"{OUT}/kmeans_2d.png")
    plot_2d_matplotlib(df, "Segment", f"{OUT}/segments_2d.png")
    plot_3d_matplotlib(df, "KMeans_Cluster", f"{OUT}/kmeans_3d.png")
    plot_3d_plotly(df, "KMeans_Cluster", f"{OUT}/clusters_3d_interactive.html")
    df.to_csv(f"{OUT}/customer_segments.csv", index=False)
    interactions.to_csv(f"{OUT}/purchase_interactions.csv")
    print("\nSaved: kmeans_2d.png, segments_2d.png, kmeans_3d.png, clusters_3d_interactive.html, customer_segments.csv, purchase_interactions.csv")

