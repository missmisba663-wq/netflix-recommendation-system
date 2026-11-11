"""Poster thumbnails from Wikipedia (no API key needed).

The Kaggle dataset has no images, so we look up each title's Wikipedia article
and take its lead image (usually the official poster / title card):

    python -m src.posters            # build data/processed/posters.csv (batched, a few minutes)

A poster is accepted only if the article title matches the Netflix title, so a
missing poster falls back to the styled tile instead of showing a wrong image.
The app reads the CSV, so the live site never calls Wikipedia at runtime.
"""
from __future__ import annotations

import re
import time
import unicodedata

import pandas as pd
import requests

from .config import PROCESSED_DIR

POSTERS_CSV = PROCESSED_DIR / "posters.csv"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "NetflixRecommenderStudentProject/1.0 (educational, non-commercial)"}
_session = requests.Session()
_session.headers.update(HEADERS)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\(.*?\)", "", s.lower())          # drop "(2019 film)"
    return re.sub(r"[^a-z0-9]+", "", s.replace("&", "and"))


def _candidates(title: str, year: int, kind: str) -> list[tuple[str, bool]]:
    """Article names to try, in priority order. bool = needs description check."""
    if kind == "Movie":
        return [(f"{title} ({year} film)", False), (f"{title} (film)", True), (title, True)]
    return [(f"{title} ({year} TV series)", False), (f"{title} (TV series)", True),
            (f"{title} (web series)", True), (title, True)]


def _desc_ok(desc: str, year: int, kind: str) -> bool:
    d = (desc or "").lower()
    if kind == "Movie":
        return "film" in d and any(str(y) in d for y in (year - 1, year, year + 1))
    return any(w in d for w in ("series", "television", "tv show", "sitcom", "anime", "drama",
                                "reality", "show"))


def _batch_lookup(names: list[str]) -> dict[str, dict] | None:
    """Resolve up to 50 article names -> {requested name: {thumb, desc}}."""
    params = {"action": "query", "format": "json", "formatversion": 2, "redirects": 1,
              "titles": "|".join(names), "prop": "pageimages|description",
              "piprop": "thumbnail", "pithumbsize": 400, "pilicense": "any", "pilimit": 50}
    for attempt in range(8):
        try:
            r = _session.get(API, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0) or 0) or 20 * (attempt + 1)
                print(f"  rate-limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            q = r.json().get("query", {})
            break
        except (requests.RequestException, ValueError):
            time.sleep(5 * (attempt + 1))
    else:
        return None  # failed after retries -> caller will retry later
    alias = {n["from"]: n["to"] for n in q.get("normalized", [])}
    redir = {n["from"]: n["to"] for n in q.get("redirects", [])}
    pages = {p["title"]: p for p in q.get("pages", []) if not p.get("missing")}
    out = {}
    for name in names:
        t = alias.get(name, name)
        t = redir.get(t, t)
        p = pages.get(t)
        if p:
            out[name] = {"thumb": p.get("thumbnail", {}).get("source"),
                         "desc": p.get("description", "")}
    return out


CACHE_JSON = PROCESSED_DIR / "posters_raw.json"


def build(df: pd.DataFrame, pause: float = 1.0) -> pd.DataFrame:
    """Sequential and resumable: raw API results are cached in posters_raw.json."""
    import json
    jobs = [(r.show_id, r.title, int(r.release_year), r.type) for r in df.itertuples()]
    cands = {sid: _candidates(t, y, k) for sid, t, y, k in jobs}
    names = sorted({n for c in cands.values() for n, _ in c})
    cache = json.loads(CACHE_JSON.read_text(encoding="utf-8")) if CACHE_JSON.exists() else {}
    found: dict[str, dict] = cache.get("found", {})
    checked = set(cache.get("checked", []))
    todo = [n for n in names if n not in checked]
    chunks = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    print(f"{len(names):,} article names, {len(todo):,} to look up in {len(chunks)} requests", flush=True)
    for i, chunk in enumerate(chunks, 1):
        res = _batch_lookup(chunk)
        if res is not None:  # {} is a valid answer: none of these 50 articles exist
            found.update(res)
            checked.update(chunk)
        if i % 20 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)} requests", flush=True)
            CACHE_JSON.write_text(json.dumps({"found": found, "checked": sorted(checked)}),
                                  encoding="utf-8")
        time.sleep(pause)
    rows = []
    for sid, t, y, k in jobs:
        url = ""
        for name, check in cands[sid]:
            hit = found.get(name)
            if not hit or not hit["thumb"] or hit["thumb"].lower().split("?")[0].endswith(".svg.png"):
                continue
            if check and not _desc_ok(hit["desc"], y, k):
                continue
            if _norm(name) != _norm(t):
                continue
            url = hit["thumb"]
            break
        rows.append({"show_id": sid, "poster_url": url})
    out = pd.DataFrame(rows)
    out.to_csv(POSTERS_CSV, index=False)
    return out


def load_posters() -> dict[str, str]:
    if not POSTERS_CSV.exists():
        return {}
    p = pd.read_csv(POSTERS_CSV, keep_default_na=False)
    return dict(zip(p["show_id"], p["poster_url"]))


if __name__ == "__main__":
    from .data_access import load_titles
    t0 = time.time()
    res = build(load_titles())
    hit = (res["poster_url"] != "").mean()
    print(f"Done: {len(res):,} titles, {hit:.0%} with posters, {time.time() - t0:.0f}s")
