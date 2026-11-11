"""Framework-free loaders shared by the Streamlit app and the FastAPI backend."""
from __future__ import annotations

import pandas as pd

from .config import CLEAN_CSV
from .data_prep import load_clean
from .recommender import NetflixRecommender


def load_titles() -> pd.DataFrame:
    if CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV, parse_dates=["date_added"], keep_default_na=False,
                         na_values={"duration_min": [""], "seasons": [""], "year_added": [""],
                                    "month_added": [""], "years_to_netflix": [""],
                                    "date_added": [""]})
    else:  # fallback: build from raw data
        df = load_clean()
    for c in ["director", "cast", "country", "genres", "countries", "cast_list", "directors"]:
        df[c] = df[c].fillna("").astype(str)
    return df


def build_recommender(df: pd.DataFrame) -> NetflixRecommender:
    return NetflixRecommender().fit(df)
