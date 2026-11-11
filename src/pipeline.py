"""End-to-end ML pipeline.

    python -m src.pipeline            # download -> clean -> train -> cluster -> evaluate -> export

Outputs
    data/processed/netflix_clean.csv     cleaned + engineered + cluster labels (used by the app)
    data/processed/cluster_summary.csv
    data/processed/evaluation.csv
    powerbi/data/*.csv                   star-schema tables for the Power BI dashboard
"""
from __future__ import annotations

import time

import pandas as pd

from .classifier import ContentClassifier
from .clustering import cluster_titles
from .config import CLEAN_CSV, CLUSTER_CSV, EVAL_CSV, POWERBI_DIR, PROCESSED_DIR
from .data_prep import download_raw, load_clean, split_list
from .evaluate import evaluate
from .recommender import NetflixRecommender


def export_powerbi(df: pd.DataFrame, clusters: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    titles = df[[
        "show_id", "title", "type", "primary_genre", "primary_country", "date_added",
        "year_added", "month_added", "month_name", "weekday_added", "release_year", "decade",
        "rating", "maturity_group", "duration_min", "seasons", "n_genres", "n_countries",
        "cast_size", "years_to_netflix", "description_len", "cluster", "cluster_name",
        "tsne_x", "tsne_y", "description",
    ]].copy()
    titles["date_added"] = titles["date_added"].dt.strftime("%Y-%m-%d")
    titles.to_csv(POWERBI_DIR / "Titles.csv", index=False)

    bridges = {
        "Title_Genres.csv": ("genres", "genre"),
        "Title_Countries.csv": ("countries", "country"),
        "Title_Cast.csv": ("cast_list", "actor"),
        "Title_Directors.csv": ("directors", "director"),
    }
    for fname, (col, out_col) in bridges.items():
        b = df[["show_id"]].assign(**{out_col: split_list(df[col])}).explode(out_col).dropna()
        b.to_csv(POWERBI_DIR / fname, index=False)
    clusters.to_csv(POWERBI_DIR / "Clusters.csv", index=False)
    evaluation.to_csv(POWERBI_DIR / "Model_Evaluation.csv", index=False)


def run(n_eval_queries: int = 400) -> None:
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    print("1/5 Downloading Kaggle dataset (if missing)...")
    download_raw()
    print("2/5 Cleaning & feature engineering...")
    df = load_clean()
    print(f"    {len(df):,} titles, {df.shape[1]} columns")
    print("3/5 Training hybrid recommender...")
    rec = NetflixRecommender().fit(df)
    print(f"    feature matrix {rec.X.shape}, nnz={rec.X.nnz:,}")
    print("4/5 Clustering (LSA + KMeans + t-SNE)...")
    clus, summary, sil = cluster_titles(rec)
    print(f"    silhouette (cosine) = {sil:.3f}")
    df = pd.concat([df, clus], axis=1)
    summary["silhouette"] = sil
    print("5/5 Evaluating vs baselines...")
    ev = evaluate(rec, n_queries=n_eval_queries)
    print(ev.round(3).to_string(index=False))

    print("+   Training supervised NLP classifiers...")
    clf = ContentClassifier().fit(df)
    print(clf.report.summary().round(3).to_string(index=False))
    clf.report.summary().to_csv(POWERBI_DIR / "Classifier_Metrics.csv", index=False)
    clf.report.per_genre.to_csv(POWERBI_DIR / "Genre_Classifier.csv", index=False)

    df.to_csv(CLEAN_CSV, index=False)
    summary.to_csv(CLUSTER_CSV, index=False)
    ev.to_csv(EVAL_CSV, index=False)
    export_powerbi(df, summary, ev)
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    run()
