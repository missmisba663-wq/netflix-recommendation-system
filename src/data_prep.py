"""Data cleaning & feature engineering for the Kaggle Netflix titles dataset.

Dataset: "Netflix Movies and TV Shows" by Shivam Bansal
https://www.kaggle.com/datasets/shivamb/netflix-shows
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import KAGGLE_DATASET, LIST_SEP, RAW_CSV

MATURITY_GROUPS = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "G": "Kids", "TV-G": "Kids",
    "PG": "Older Kids", "TV-PG": "Older Kids",
    "PG-13": "Teens", "TV-14": "Teens",
    "R": "Adults", "NC-17": "Adults", "TV-MA": "Adults",
    "NR": "Unrated", "UR": "Unrated",
}


def download_raw(force: bool = False) -> None:
    """Fetch the dataset from Kaggle (public dataset, via kagglehub)."""
    if RAW_CSV.exists() and not force:
        return
    import shutil

    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{path}/netflix_titles.csv", RAW_CSV)


def _split(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df[c] = df[c].str.strip()

    # Known data issue: for 3 titles the duration was shifted into `rating`.
    shifted = df["rating"].str.contains("min", na=False)
    df.loc[shifted & df["duration"].isna(), "duration"] = df.loc[shifted, "rating"]
    df.loc[shifted, "rating"] = pd.NA
    df["rating"] = df["rating"].fillna("NR")

    df["date_added"] = pd.to_datetime(df["date_added"], format="%B %d, %Y", errors="coerce")
    for c in ["director", "cast", "country"]:
        df[c] = df[c].fillna("")
    df["duration"] = df["duration"].fillna("")
    df = df.drop_duplicates(subset=["title", "type", "release_year"]).reset_index(drop=True)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    genres = [_split(v) for v in df["listed_in"].tolist()]
    countries = [_split(v) for v in df["country"].tolist()]
    cast = [_split(v) for v in df["cast"].tolist()]
    directors = [_split(v) for v in df["director"].tolist()]

    df["genres"] = [LIST_SEP.join(x) for x in genres]
    df["countries"] = [LIST_SEP.join(x) for x in countries]
    df["cast_list"] = [LIST_SEP.join(x) for x in cast]
    df["directors"] = [LIST_SEP.join(x) for x in directors]
    df["primary_genre"] = [g[0] if g else "Unknown" for g in genres]
    df["primary_country"] = [c[0] if c else "Unknown" for c in countries]
    df["n_genres"] = [len(x) for x in genres]
    df["n_countries"] = [len(x) for x in countries]
    df["cast_size"] = [len(x) for x in cast]

    dur_num = pd.to_numeric(df["duration"].str.extract(r"(\d+)")[0], errors="coerce")
    is_movie = df["type"].eq("Movie")
    df["duration_min"] = np.where(is_movie, dur_num, np.nan)
    df["seasons"] = np.where(~is_movie, dur_num, np.nan)

    df["year_added"] = df["date_added"].dt.year.astype("Int64")
    df["month_added"] = df["date_added"].dt.month.astype("Int64")
    df["month_name"] = df["date_added"].dt.month_name()
    df["weekday_added"] = df["date_added"].dt.day_name()
    df["years_to_netflix"] = (df["year_added"] - df["release_year"]).clip(lower=0)
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
    df["maturity_group"] = df["rating"].map(MATURITY_GROUPS).fillna("Unrated")
    df["is_original_era"] = df["release_year"] >= 2015  # Netflix originals boom
    df["description_len"] = df["description"].str.split().str.len()
    df["label"] = df["title"] + " (" + df["release_year"].astype(str) + ", " + df["type"] + ")"
    return df


def load_clean(raw_path=RAW_CSV) -> pd.DataFrame:
    return engineer(clean(pd.read_csv(raw_path)))


def split_list(series: pd.Series) -> pd.Series:
    """Turn a pipe-separated column back into python lists."""
    return pd.Series([[x for x in str(s).split(LIST_SEP) if x] for s in series.fillna("")],
                     index=series.index, dtype=object)
