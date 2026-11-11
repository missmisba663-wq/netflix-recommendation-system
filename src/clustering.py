"""Unsupervised content clustering (LSA + K-Means) and 2-D map (t-SNE)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .config import LIST_SEP, N_CLUSTERS, RANDOM_STATE
from .recommender import NetflixRecommender


GENERIC = ["Movies", "TV Shows", "International Movies", "International TV Shows"]


def cluster_titles(rec: NetflixRecommender, k: int = N_CLUSTERS):
    df = rec.df
    svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
    Z = normalize(svd.fit_transform(rec.X))
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(Z)
    sample = np.random.default_rng(RANDOM_STATE).choice(len(Z), size=min(3000, len(Z)), replace=False)
    sil = float(silhouette_score(Z[sample], labels[sample], metric="cosine"))

    xy = TSNE(n_components=2, perplexity=40, init="pca", random_state=RANDOM_STATE,
              metric="cosine").fit_transform(Z[:, :50])

    # Name each cluster by its most over-represented genres (lift vs catalogue).
    exploded = df.assign(cluster=labels, g=df["genres"].str.split(LIST_SEP)).explode("g")
    base = exploded["g"].value_counts(normalize=True)
    vocab = rec.vectorizers["description"].get_feature_names_out()
    D = rec.blocks["description"]
    summary = []
    for c in range(k):
        sub = exploded[exploded["cluster"] == c]
        share = sub["g"].value_counts(normalize=True)
        counts = sub["g"].value_counts()
        lift = share / base.reindex(share.index)
        # distinctive AND common: share x sqrt(lift), ignoring catch-all categories
        score = (share * np.sqrt(lift)).where(counts >= 15).drop(GENERIC, errors="ignore").dropna()
        top_g = score.sort_values(ascending=False).head(2).index.tolist() or share.head(2).index.tolist()
        members = np.where(labels == c)[0]
        centroid = np.asarray(D[members].mean(axis=0)).ravel()
        terms = [vocab[i] for i in np.argsort(-centroid)[:8]]
        summary.append({
            "cluster": c,
            "cluster_name": " / ".join(g.replace(" TV Shows", " TV").replace(" Movies", "") for g in top_g),
            "size": len(members),
            "pct_movies": float((df["type"].iloc[members] == "Movie").mean()),
            "median_year": int(df["release_year"].iloc[members].median()),
            "top_genres": ", ".join(share.head(4).index),
            "top_terms": ", ".join(terms),
        })
    summary = pd.DataFrame(summary)
    names = dict(zip(summary["cluster"], summary["cluster_name"]))
    out = pd.DataFrame({
        "cluster": labels,
        "cluster_name": [names[l] for l in labels],
        "tsne_x": xy[:, 0],
        "tsne_y": xy[:, 1],
    })
    return out, summary, sil
