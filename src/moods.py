"""Mood-based discovery: each mood = keyword profile + preferred genres.

score = 0.55 * keyword relevance (TF-IDF cosine vs mood keywords)
      + 0.45 * genre affinity  (share of the title's genres that fit the mood)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .config import LIST_SEP
from .recommender import Filters, NetflixRecommender

MOODS: dict[str, dict] = {
    "😊 Feel-good": {
        "words": "heartwarming uplifting friendship love family joy hope funny charming kindness dream",
        "genres": ["Comedies", "Romantic Movies", "Children & Family Movies", "TV Comedies",
                   "Romantic TV Shows", "Music & Musicals"]},
    "🌑 Dark & gritty": {
        "words": "crime murder drug cartel gang corrupt revenge violent killer dark underworld mafia",
        "genres": ["Crime TV Shows", "Thrillers", "TV Thrillers", "Dramas", "TV Dramas"]},
    "💘 Romantic": {
        "words": "love romance falls in love relationship wedding heart couple passion",
        "genres": ["Romantic Movies", "Romantic TV Shows"]},
    "🧠 Mind-bending": {
        "words": "mystery time travel secret alternate reality strange conspiracy puzzle mind memory",
        "genres": ["TV Mysteries", "Sci-Fi & Fantasy", "TV Sci-Fi & Fantasy", "Thrillers",
                   "Cult Movies"]},
    "⚡ Adrenaline": {
        "words": "action mission explosive chase fight soldier heist battle survive race assassin",
        "genres": ["Action & Adventure", "TV Action & Adventure", "Thrillers"]},
    "👻 Scary": {
        "words": "horror haunted demon ghost evil terrifying possessed curse supernatural nightmare",
        "genres": ["Horror Movies", "TV Horror"]},
    "🌍 Curious / learn something": {
        "words": "documentary explores history science nature investigation true story world planet",
        "genres": ["Documentaries", "Docuseries", "Science & Nature TV"]},
    "😂 Laugh out loud": {
        "words": "comedian stand-up hilarious comedy jokes funny special laugh",
        "genres": ["Stand-Up Comedy", "Stand-Up Comedy & Talk Shows", "Comedies", "TV Comedies"]},
    "👨‍👩‍👧 Family night": {
        "words": "family kids adventure animated friends magical journey fun young",
        "genres": ["Children & Family Movies", "Kids' TV"]},
    "🏆 Inspiring": {
        "words": "true story overcome dream champion inspiring journey triumph courage athlete",
        "genres": ["Sports Movies", "Documentaries", "Dramas", "Docuseries"]},
}


def mood_recommend(rec: NetflixRecommender, mood: str, k: int = 12,
                   filters: Filters | None = None, diversity: float = 0.3) -> pd.DataFrame:
    spec = MOODS[mood]
    q = normalize(rec.soup_vec.transform([spec["words"]]))
    kw = (rec.soup_X @ q.T).toarray().ravel()
    kw = kw / (kw.max() or 1.0)
    want = set(spec["genres"])
    aff = np.array([
        (len(want & set(gs)) / len(gs)) if gs else 0.0
        for gs in (str(s).split(LIST_SEP) for s in rec.df["genres"])
    ])
    scores = 0.55 * kw + 0.45 * aff
    ranked = rec._rank(scores, k, set(), filters, diversity)
    return rec._to_frame(ranked)


def neighbour_graph(rec: NetflixRecommender, show_id: str, k1: int = 8, k2: int = 3):
    """Two-hop similarity graph around a title -> (nodes DataFrame, edges list)."""
    nodes = {show_id: {"level": 0, "parent": None, "score": 1.0}}
    edges = []
    seen: set[frozenset] = set()

    def link(a, b, w):
        key = frozenset((a, b))
        if a != b and key not in seen:
            seen.add(key)
            edges.append((a, b, w))

    first = rec.similar(show_id, k=k1, explain=False)
    for _, r in first.iterrows():
        nodes[r["show_id"]] = {"level": 1, "parent": show_id, "score": r["score"]}
        link(show_id, r["show_id"], r["score"])
    for sid in list(first["show_id"]):
        second = rec.similar(sid, k=k2 + k1, explain=False)
        added = cross = 0
        for _, r in second.iterrows():
            if added == k2:
                break
            if r["show_id"] in nodes:
                if cross < 2 and r["show_id"] != show_id:
                    link(sid, r["show_id"], r["score"])  # cross-link between neighbours
                    cross += 1
                continue
            nodes[r["show_id"]] = {"level": 2, "parent": sid, "score": r["score"]}
            link(sid, r["show_id"], r["score"])
            added += 1
    nd = pd.DataFrame.from_dict(nodes, orient="index")
    nd.index.name = "show_id"
    nd = nd.reset_index().merge(rec.df[["show_id", "title", "type", "release_year", "genres"]],
                                on="show_id")
    return nd, edges
