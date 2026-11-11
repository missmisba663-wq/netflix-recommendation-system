"""Offline evaluation of the recommender against baselines.

The Kaggle dataset has no user ratings, so we use standard *proxy* metrics for
content-based recommenders, averaged over a random sample of query titles:

* Genre Precision@K  - share of recommendations sharing >=1 genre with the query
* Genre Jaccard@K    - mean Jaccard overlap of genre sets
* People Hit@K       - share of recommendations sharing a director or cast member
* Plot Similarity@K - mean description (TF-IDF) cosine between query and recs
* Type Match@K       - share with the same type (Movie / TV Show)
* Intra-list Diversity - 1 - mean pairwise description similarity inside the list
* Catalog Coverage   - % of the catalogue that appears in any recommendation list
* Novelty            - mean -log2(popularity) of recommended genres (higher = less obvious)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import LIST_SEP, RANDOM_STATE
from .recommender import NetflixRecommender


def _sets(series: pd.Series) -> list[set]:
    return [set(x for x in str(s).split(LIST_SEP) if x) for s in series.fillna("")]


def evaluate(rec: NetflixRecommender, n_queries: int = 400, k: int = 10) -> pd.DataFrame:
    df = rec.df
    rng = np.random.default_rng(RANDOM_STATE)
    queries = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
    genres = _sets(df["genres"])
    people = [a | b for a, b in zip(_sets(df["directors"]), _sets(df["cast_list"]))]
    types = df["type"].to_numpy()
    D = rec.blocks["description"]
    genre_counts = pd.Series([g for gs in genres for g in gs]).value_counts()
    genre_pop = (genre_counts / len(df)).to_dict()

    original_weights = dict(rec.weights)
    methods = {
        "Random baseline": None,
        "Genre-only": {"genres": 1},
        "Description-only (TF-IDF)": {"description": 1},
        "Hybrid weighted (ours)": original_weights,
        "Hybrid + MMR diversity (ours)": original_weights,
    }
    rows = []
    for name, weights in methods.items():
        t0 = time.perf_counter()
        diversity = 0.3 if "MMR" in name else 0.0
        stats = {"gp": [], "gj": [], "ph": [], "tm": [], "div": [], "nov": [], "plot": []}
        seen: set[int] = set()
        for q in queries:
            if weights is None:
                recs = rng.choice(np.delete(np.arange(len(df)), q), size=k, replace=False)
            else:
                out = rec.similar(df["show_id"].iat[q], k=k, diversity=diversity,
                                  weights=weights, explain=False)
                recs = np.array([rec.idx(s) for s in out["show_id"]])
            seen.update(recs.tolist())
            gq = genres[q]
            stats["gp"].append(np.mean([bool(gq & genres[r]) for r in recs]))
            stats["gj"].append(np.mean([len(gq & genres[r]) / max(len(gq | genres[r]), 1) for r in recs]))
            stats["ph"].append(np.mean([bool(people[q] & people[r]) for r in recs]))
            stats["tm"].append(np.mean([types[q] == types[r] for r in recs]))
            stats["plot"].append((D[recs] @ D[q].T).toarray().mean())
            sims = (D[recs] @ D[recs].T).toarray()
            iu = np.triu_indices(len(recs), 1)
            stats["div"].append(1 - sims[iu].mean())
            stats["nov"].append(np.mean([
                np.mean([-np.log2(genre_pop.get(g, 1e-3)) for g in genres[r]] or [0]) for r in recs
            ]))
        elapsed = (time.perf_counter() - t0) / len(queries) * 1000
        rows.append({
            "Method": name,
            "Genre Precision@10": np.mean(stats["gp"]),
            "Genre Jaccard@10": np.mean(stats["gj"]),
            "People Hit@10": np.mean(stats["ph"]),
            "Plot Similarity@10": np.mean(stats["plot"]),
            "Type Match@10": np.mean(stats["tm"]),
            "Intra-list Diversity": np.mean(stats["div"]),
            "Novelty": np.mean(stats["nov"]),
            "Catalog Coverage": len(seen) / len(df),
            "Latency (ms/query)": elapsed,
        })
    return pd.DataFrame(rows)
