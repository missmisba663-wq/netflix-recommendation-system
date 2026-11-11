"""Hybrid content-based recommendation engine.

Model
-----
Every title is represented by several feature blocks, each vectorised with
TF-IDF and L2-normalised:

    description  - plot synopsis (words + bigrams, English stop-words removed)
    genres       - Netflix "listed_in" categories (one token per genre)
    cast         - actors (one token per person)
    director     - directors (one token per person)
    country      - production countries
    meta         - type + maturity rating + release decade

Each block is scaled by sqrt(w_f) and the blocks are horizontally stacked, so
the dot product of two title vectors equals the *weighted sum of per-feature
cosine similarities*:

    sim(a, b) = sum_f  w_f * cos(a_f, b_f)          (sum w_f = 1, sim in [0, 1])

This keeps scores interpretable (we can show *why* something was recommended)
and avoids storing an 8.8k x 8.8k similarity matrix - similarities are computed
on demand with one sparse mat-vec (~ms).

Extra modes
-----------
* Taste profile (Rocchio): user vector = mean(liked) - beta * mean(disliked)
* Free-text search: TF-IDF over a "soup" of all text fields
* Diversity: Maximal Marginal Relevance (MMR) re-ranking
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import FEATURE_WEIGHTS, LIST_SEP


# Query understanding: nationality / industry words -> dataset country names.
DEMONYMS = {
    "korean": "South Korea", "k-drama": "South Korea", "kdrama": "South Korea",
    "indian": "India", "bollywood": "India", "hindi": "India", "tamil": "India", "telugu": "India",
    "japanese": "Japan", "anime": "Japan", "british": "United Kingdom", "uk": "United Kingdom",
    "american": "United States", "hollywood": "United States", "spanish": "Spain",
    "mexican": "Mexico", "french": "France", "german": "Germany", "turkish": "Turkey",
    "nigerian": "Nigeria", "nollywood": "Nigeria", "egyptian": "Egypt", "chinese": "China",
    "taiwanese": "Taiwan", "thai": "Thailand", "brazilian": "Brazil", "italian": "Italy",
    "canadian": "Canada", "australian": "Australia", "filipino": "Philippines",
    "pakistani": "Pakistan", "argentinian": "Argentina", "colombian": "Colombia",
}


def query_countries(text: str) -> set[str]:
    words = text.lower().replace(",", " ").split()
    return {DEMONYMS[w] for w in words if w in DEMONYMS}


def _pipe_tokenizer(text: str) -> list[str]:
    return [t for t in text.split(LIST_SEP) if t]


def _entity_vectorizer() -> TfidfVectorizer:
    # Each person / genre / country is a single token.
    return TfidfVectorizer(tokenizer=_pipe_tokenizer, token_pattern=None, lowercase=True)


@dataclass
class Filters:
    content_type: str | None = None          # "Movie" / "TV Show"
    min_year: int | None = None
    max_year: int | None = None
    genres: list[str] = field(default_factory=list)       # match ANY
    maturity: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        if self.content_type:
            m &= (df["type"] == self.content_type).to_numpy()
        if self.min_year:
            m &= (df["release_year"] >= self.min_year).to_numpy()
        if self.max_year:
            m &= (df["release_year"] <= self.max_year).to_numpy()
        if self.genres:
            gs = set(self.genres)
            m &= df["genres"].fillna("").map(lambda s: bool(gs & set(s.split(LIST_SEP)))).to_numpy()
        if self.maturity:
            m &= df["maturity_group"].isin(self.maturity).to_numpy()
        if self.countries:
            cs = set(self.countries)
            m &= df["countries"].fillna("").map(lambda s: bool(cs & set(s.split(LIST_SEP)))).to_numpy()
        return m


class NetflixRecommender:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or FEATURE_WEIGHTS)
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.blocks: dict[str, sparse.csr_matrix] = {}

    # ------------------------------------------------------------------ fit
    def _block_texts(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        meta = (
            "type_" + df["type"].str.replace(" ", "")
            + LIST_SEP + "rating_" + df["rating"].astype(str)
            + LIST_SEP + "decade_" + df["decade"].astype(str)
        )
        return {
            "description": df["description"].fillna(""),
            "genres": df["genres"].fillna(""),
            "cast": df["cast_list"].fillna(""),
            "director": df["directors"].fillna(""),
            "country": df["countries"].fillna(""),
            "meta": meta,
        }

    def fit(self, df: pd.DataFrame) -> "NetflixRecommender":
        self.df = df.reset_index(drop=True)
        self.id_to_idx = {sid: i for i, sid in enumerate(self.df["show_id"])}
        texts = self._block_texts(self.df)
        for name, text in texts.items():
            if name == "description":
                vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                                      max_df=0.5, sublinear_tf=True)
            else:
                vec = _entity_vectorizer()
            X = normalize(vec.fit_transform(text)).tocsr().astype(np.float32)
            self.vectorizers[name] = vec
            self.blocks[name] = X
        self._rebuild_matrix()

        soup = (
            self.df["title"] + " " + self.df["description"] + " "
            + self.df["listed_in"].fillna("") + " " + self.df["country"] + " "
            + self.df["director"] + " " + self.df["cast"] + " " + self.df["type"]
        )
        self.soup_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                        min_df=1, sublinear_tf=True)
        self.soup_X = normalize(self.soup_vec.fit_transform(soup)).tocsr().astype(np.float32)
        # character n-grams are robust to word forms (romance/romantic, zombie/zombies)
        short = self.df["title"] + " " + self.df["listed_in"].fillna("") + " " + self.df["description"]
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                        sublinear_tf=True)
        self.char_X = normalize(self.char_vec.fit_transform(short)).tocsr().astype(np.float32)
        return self

    def _rebuild_matrix(self) -> None:
        parts = [self.blocks[n] * np.float32(np.sqrt(w)) for n, w in self.weights.items()]
        self.X = sparse.hstack(parts).tocsr()

    def set_weights(self, weights: dict[str, float]) -> None:
        total = sum(weights.values()) or 1.0
        self.weights = {k: weights.get(k, 0) / total for k in self.blocks}
        self._rebuild_matrix()

    # ----------------------------------------------------------- helpers
    def idx(self, show_id: str) -> int:
        return self.id_to_idx[show_id]

    def find(self, query: str, limit: int = 10) -> pd.DataFrame:
        q = query.lower().strip()
        titles = self.df["title"].str.lower()
        hits = self.df[titles.str.contains(q, regex=False, na=False)].copy()
        hits["_rank"] = (~titles[hits.index].str.startswith(q)).astype(int)
        return hits.sort_values(["_rank", "release_year"], ascending=[True, False]).head(limit)

    def _rank(self, scores: np.ndarray, k: int, exclude: set[int], filters: Filters | None,
              diversity: float) -> list[tuple[int, float]]:
        scores = scores.astype(np.float64).copy()
        if filters:
            scores[~filters.mask(self.df)] = -np.inf
        for i in exclude:
            scores[i] = -np.inf
        n_cand = k if diversity <= 0 else max(k * 8, 60)
        cand = np.argpartition(-scores, min(n_cand, len(scores) - 1))[:n_cand]
        cand = cand[np.isfinite(scores[cand])]
        cand = cand[np.argsort(-scores[cand])]
        if diversity <= 0 or len(cand) <= k:
            return [(int(i), float(scores[i])) for i in cand[:k]]
        return self._mmr(cand, scores, k, diversity)

    def _mmr(self, cand: np.ndarray, scores: np.ndarray, k: int, diversity: float):
        """Maximal Marginal Relevance: lambda*rel - (1-lambda)*max_sim_to_selected."""
        lam = 1.0 - diversity
        Xc = self.X[cand]
        pair = (Xc @ Xc.T).toarray()
        rel = scores[cand]
        selected: list[int] = []
        remaining = list(range(len(cand)))
        while remaining and len(selected) < k:
            if selected:
                red = pair[np.ix_(remaining, selected)].max(axis=1)
            else:
                red = np.zeros(len(remaining))
            mmr = lam * rel[remaining] - (1 - lam) * red
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)
        return [(int(cand[i]), float(rel[i])) for i in selected]

    def _to_frame(self, ranked: list[tuple[int, float]]) -> pd.DataFrame:
        if not ranked:
            return self.df.head(0).assign(score=[])
        idxs = [i for i, _ in ranked]
        out = self.df.iloc[idxs].copy()
        out["score"] = [s for _, s in ranked]
        return out

    def normalize_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        if not weights:
            return self.weights
        total = sum(max(v, 0) for v in weights.values()) or 1.0
        return {k: max(weights.get(k, 0), 0) / total for k in self.blocks}

    # ------------------------------------------------------- public API
    def similar(self, show_id: str, k: int = 10, filters: Filters | None = None,
                diversity: float = 0.0, weights: dict[str, float] | None = None,
                explain: bool = True) -> pd.DataFrame:
        """Titles most similar to one title (item-to-item).

        `weights` overrides the feature weights for this call only (thread-safe:
        the shared model is never mutated). Output includes the per-feature
        contribution `sim_<feature>` so the score can be explained.
        """
        i = self.idx(show_id)
        w = self.normalize_weights(weights)
        per = {n: (B @ B[i].T).toarray().ravel() for n, B in self.blocks.items()}
        scores = sum(w[n] * per[n] for n in self.blocks)
        ranked = self._rank(scores, k, {i}, filters, diversity)
        out = self._to_frame(ranked)
        idxs = [j for j, _ in ranked]
        for n in self.blocks:
            out[f"sim_{n}"] = w[n] * per[n][idxs] if idxs else []
        if explain:
            out["reason"] = [self.explain(show_id, sid) for sid in out["show_id"]]
        return out

    def profile(self, liked: list[str], disliked: list[str] | None = None, k: int = 10,
                filters: Filters | None = None, diversity: float = 0.0,
                beta: float = 0.5) -> pd.DataFrame:
        """Rocchio taste-profile recommendations from several liked titles."""
        liked_idx = [self.idx(s) for s in liked if s in self.id_to_idx]
        dis_idx = [self.idx(s) for s in (disliked or []) if s in self.id_to_idx]
        if not liked_idx:
            return self.df.head(0).assign(score=[])
        vec = np.asarray(self.X[liked_idx].mean(axis=0)).ravel()
        if dis_idx:
            vec -= beta * np.asarray(self.X[dis_idx].mean(axis=0)).ravel()
        vec /= np.linalg.norm(vec) or 1.0
        scores = self.X @ vec
        ranked = self._rank(np.asarray(scores).ravel(), k, set(liked_idx) | set(dis_idx),
                            filters, diversity)
        return self._to_frame(ranked)

    def search(self, text: str, k: int = 10, filters: Filters | None = None,
               diversity: float = 0.0) -> pd.DataFrame:
        """Natural-language / keyword search ("korean crime thriller").

        score = 0.6 * word TF-IDF + 0.4 * char n-gram TF-IDF (+0.15 if the title is
        from a country implied by the query, e.g. "korean" -> South Korea).
        """
        word = (self.soup_X @ normalize(self.soup_vec.transform([text])).T).toarray().ravel()
        char = (self.char_X @ normalize(self.char_vec.transform([text])).T).toarray().ravel()
        scores = 0.6 * word + 0.4 * char
        if countries := query_countries(text):
            in_country = self.df["countries"].map(
                lambda c: bool(countries & set(str(c).split(LIST_SEP)))).to_numpy()
            scores = scores + 0.15 * in_country
        ranked = self._rank(scores, k, set(), filters, diversity)
        return self._to_frame([(i, s) for i, s in ranked if s > 0])

    def explain(self, source_id: str, target_id: str) -> str:
        a = self.df.iloc[self.idx(source_id)]
        b = self.df.iloc[self.idx(target_id)]
        parts = []

        def shared(col):
            sa = {x for x in str(a[col]).split(LIST_SEP) if x}
            sb = {x for x in str(b[col]).split(LIST_SEP) if x}
            return sorted(sa & sb)

        if d := shared("directors"):
            parts.append("Same director: " + ", ".join(d[:2]))
        if c := shared("cast_list"):
            parts.append("Shared cast: " + ", ".join(c[:3]))
        if g := shared("genres"):
            parts.append("Genres: " + ", ".join(g[:3]))
        if co := shared("countries"):
            parts.append("From " + ", ".join(co[:2]))
        da = (self.blocks["description"][self.idx(source_id)]
              .multiply(self.blocks["description"][self.idx(target_id)])).tocsr()
        if da.nnz:
            vocab = self.vectorizers["description"].get_feature_names_out()
            words, used = [], set()
            for t in np.argsort(-da.data):
                term = vocab[da.indices[t]]
                if set(term.split()) & used:
                    continue
                words.append(term)
                used.update(term.split())
                if len(words) == 4:
                    break
            parts.append("Similar plot themes: " + ", ".join(words))
        return " · ".join(parts) if parts else "Similar overall content profile"
