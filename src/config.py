"""Central paths and constants for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "netflix_titles.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV = PROCESSED_DIR / "netflix_clean.csv"
EVAL_CSV = PROCESSED_DIR / "evaluation.csv"
CLUSTER_CSV = PROCESSED_DIR / "cluster_summary.csv"
DB_PATH = DATA_DIR / "app.db"
POWERBI_DIR = ROOT / "powerbi" / "data"

KAGGLE_DATASET = "shivamb/netflix-shows"
LIST_SEP = "|"  # separator used for multi-valued columns in processed files

# Feature weights for the hybrid content model (must sum to 1.0).
FEATURE_WEIGHTS = {
    "description": 0.40,
    "genres": 0.25,
    "cast": 0.15,
    "director": 0.10,
    "country": 0.05,
    "meta": 0.05,  # type + maturity rating + decade
}

N_CLUSTERS = 12
RANDOM_STATE = 42

NETFLIX_RED = "#E50914"
