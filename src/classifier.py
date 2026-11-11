"""Supervised NLP models: predict genres, type and audience from a plot synopsis.

Models (all TF-IDF features of title + description):
* Genre tagger    - multi-label, One-vs-Rest Logistic Regression, class-balanced,
                    with a per-genre decision threshold tuned on a validation split
* Type classifier - Movie vs TV Show, Logistic Regression
* Audience model  - Kids / Older Kids / Teens / Adults, multinomial Logistic Regression

Data split: 70% train / 15% validation (threshold tuning) / 15% test (reported metrics).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, hamming_loss,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .config import LIST_SEP, RANDOM_STATE

MIN_GENRE_COUNT = 60
AUDIENCES = ["Kids", "Older Kids", "Teens", "Adults"]


@dataclass
class ClassifierReport:
    genre_micro_f1: float = 0.0
    genre_macro_f1: float = 0.0
    genre_samples_f1: float = 0.0
    genre_hamming: float = 0.0
    genre_baseline_micro_f1: float = 0.0
    per_genre: pd.DataFrame = field(default_factory=pd.DataFrame)
    type_accuracy: float = 0.0
    type_f1: float = 0.0
    type_baseline: float = 0.0
    type_cm: np.ndarray | None = None
    audience_accuracy: float = 0.0
    audience_macro_f1: float = 0.0
    audience_baseline: float = 0.0
    audience_cm: np.ndarray | None = None
    n_train: int = 0
    n_test: int = 0

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Task": "Genre tagging (multi-label)", "Metric": "Micro-F1",
             "Model": self.genre_micro_f1, "Baseline": self.genre_baseline_micro_f1},
            {"Task": "Genre tagging (multi-label)", "Metric": "Macro-F1",
             "Model": self.genre_macro_f1, "Baseline": np.nan},
            {"Task": "Movie vs TV Show", "Metric": "Accuracy",
             "Model": self.type_accuracy, "Baseline": self.type_baseline},
            {"Task": "Audience (4 classes)", "Metric": "Accuracy",
             "Model": self.audience_accuracy, "Baseline": self.audience_baseline},
            {"Task": "Audience (4 classes)", "Metric": "Macro-F1",
             "Model": self.audience_macro_f1, "Baseline": np.nan},
        ])


class ContentClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                                   max_df=0.5, sublinear_tf=True, max_features=40000)

    @staticmethod
    def _text(df: pd.DataFrame) -> pd.Series:
        return df["title"].fillna("") + ". " + df["description"].fillna("")

    def fit(self, df: pd.DataFrame) -> "ContentClassifier":
        df = df.reset_index(drop=True)
        genres = [[g for g in str(s).split(LIST_SEP) if g] for s in df["genres"]]
        counts = pd.Series([g for gs in genres for g in gs]).value_counts()
        self.labels = sorted(counts[counts >= MIN_GENRE_COUNT].index)
        self.mlb = MultiLabelBinarizer(classes=self.labels)
        Y = self.mlb.fit_transform([[g for g in gs if g in self.labels] for gs in genres])

        idx = np.arange(len(df))
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE)
        va, te = train_test_split(tmp, test_size=0.50, random_state=RANDOM_STATE)
        X = self.vec.fit_transform(self._text(df))  # unsupervised vocabulary only
        rep = ClassifierReport(n_train=len(tr), n_test=len(te))

        # ---- genres: fit on train, tune per-label thresholds on validation
        self.genre_model = OneVsRestClassifier(
            LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear"))
        self.genre_model.fit(X[tr], Y[tr])
        P_va = self.genre_model.predict_proba(X[va])
        grid = np.linspace(0.2, 0.9, 15)
        self.thresholds = np.array([
            grid[np.argmax([f1_score(Y[va][:, j], P_va[:, j] >= t, zero_division=0) for t in grid])]
            for j in range(Y.shape[1])
        ])
        P_te = self.genre_model.predict_proba(X[te])
        pred = (P_te >= self.thresholds).astype(int)
        # always output at least the single most likely genre
        empty = pred.sum(axis=1) == 0
        pred[empty, P_te[empty].argmax(axis=1)] = 1
        rep.genre_micro_f1 = f1_score(Y[te], pred, average="micro", zero_division=0)
        rep.genre_macro_f1 = f1_score(Y[te], pred, average="macro", zero_division=0)
        rep.genre_samples_f1 = f1_score(Y[te], pred, average="samples", zero_division=0)
        rep.genre_hamming = hamming_loss(Y[te], pred)
        # baseline: always predict the 2 most frequent training genres
        top2 = np.argsort(-Y[tr].sum(axis=0))[:2]
        base = np.zeros_like(Y[te]); base[:, top2] = 1
        rep.genre_baseline_micro_f1 = f1_score(Y[te], base, average="micro", zero_division=0)
        rep.per_genre = pd.DataFrame({
            "Genre": self.labels,
            "Precision": precision_score(Y[te], pred, average=None, zero_division=0),
            "Recall": recall_score(Y[te], pred, average=None, zero_division=0),
            "F1": f1_score(Y[te], pred, average=None, zero_division=0),
            "Test support": Y[te].sum(axis=0),
        }).sort_values("F1", ascending=False)

        # ---- type
        t = df["type"].to_numpy()
        self.type_model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
        self.type_model.fit(X[tr], t[tr])
        tp = self.type_model.predict(X[te])
        rep.type_accuracy = accuracy_score(t[te], tp)
        rep.type_f1 = f1_score(t[te], tp, pos_label="TV Show")
        rep.type_baseline = (t[te] == pd.Series(t[tr]).mode()[0]).mean()
        rep.type_cm = confusion_matrix(t[te], tp, labels=["Movie", "TV Show"])

        # ---- audience
        a = df["maturity_group"].to_numpy()
        ok = np.isin(a, AUDIENCES)
        tr_a, te_a = tr[ok[tr]], te[ok[te]]
        self.aud_model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
        self.aud_model.fit(X[tr_a], a[tr_a])
        ap = self.aud_model.predict(X[te_a])
        rep.audience_accuracy = accuracy_score(a[te_a], ap)
        rep.audience_macro_f1 = f1_score(a[te_a], ap, average="macro")
        rep.audience_baseline = (a[te_a] == pd.Series(a[tr_a]).mode()[0]).mean()
        rep.audience_cm = confusion_matrix(a[te_a], ap, labels=AUDIENCES)

        self.report = rep
        self.vocab = self.vec.get_feature_names_out()
        return self

    # ----------------------------------------------------------- inference
    def _top_terms(self, x, coef: np.ndarray, n: int = 6) -> list[str]:
        x = x.tocsr()
        contrib = x.data * coef[x.indices]
        order = np.argsort(-contrib)
        return [self.vocab[x.indices[i]] for i in order[:n] if contrib[i] > 0]

    def predict(self, title: str, description: str) -> dict:
        x = self.vec.transform([f"{title}. {description}"])
        if x.nnz == 0:
            return {"genres": pd.DataFrame(columns=["Genre", "Probability", "Predicted", "Key words"]),
                    "type": {}, "audience": {}, "known_terms": 0}
        probs = self.genre_model.predict_proba(x)[0]
        pred = probs >= self.thresholds
        if not pred.any():
            pred[probs.argmax()] = True
        genres = pd.DataFrame({
            "Genre": self.labels, "Probability": probs, "Predicted": pred,
            "Key words": [", ".join(self._top_terms(x, est.coef_.ravel(), 5))
                          for est in self.genre_model.estimators_],
        }).sort_values("Probability", ascending=False)

        tprob = dict(zip(self.type_model.classes_, self.type_model.predict_proba(x)[0]))
        aprob = dict(zip(self.aud_model.classes_, self.aud_model.predict_proba(x)[0]))
        return {"genres": genres, "type": tprob, "audience": aprob, "known_terms": int(x.nnz)}
