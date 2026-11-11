"""FastAPI REST backend for the Netflix recommender.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import db
from src.data_access import build_recommender, load_titles
from src.recommender import Filters

app = FastAPI(title="Netflix Recommendation API", version="1.0.0",
              description="Hybrid content-based recommender trained on the Kaggle Netflix dataset.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FIELDS = ["show_id", "title", "type", "release_year", "rating", "duration", "listed_in",
          "country", "director", "description"]


@lru_cache(maxsize=1)
def model():
    return build_recommender(load_titles())


def records(df: pd.DataFrame, extra: list[str] | None = None) -> list[dict]:
    cols = FIELDS + [c for c in (extra or []) if c in df.columns]
    out = df[cols].copy()
    if "score" in out:
        out["score"] = out["score"].round(4)
    return out.to_dict(orient="records")


class ProfileRequest(BaseModel):
    liked: list[str] = Field(..., min_length=1, examples=[["s2", "s5"]])
    disliked: list[str] = []
    k: int = Field(10, ge=1, le=50)
    content_type: str | None = None
    diversity: float = Field(0.0, ge=0, le=0.95)


class WatchItem(BaseModel):
    show_id: str


@app.get("/health")
def health():
    return {"status": "ok", "titles": len(model().df)}


@app.get("/titles")
def titles(q: str = Query("", description="Title substring"), limit: int = Query(20, le=100)):
    rec = model()
    df = rec.find(q, limit) if q else rec.df.head(limit)
    return records(df)


@app.get("/titles/{show_id}")
def title(show_id: str):
    rec = model()
    if show_id not in rec.id_to_idx:
        raise HTTPException(404, "Title not found")
    return records(rec.df.iloc[[rec.idx(show_id)]])[0]


@app.get("/recommend/{show_id}")
def recommend(show_id: str, k: int = Query(10, ge=1, le=50), content_type: str | None = None,
              diversity: float = Query(0.0, ge=0, le=0.95)):
    rec = model()
    if show_id not in rec.id_to_idx:
        raise HTTPException(404, "Title not found")
    out = rec.similar(show_id, k=k, filters=Filters(content_type=content_type), diversity=diversity)
    return {"source": rec.df.iloc[rec.idx(show_id)]["title"],
            "recommendations": records(out, ["score", "reason"])}


@app.post("/recommend/profile")
def recommend_profile(req: ProfileRequest):
    rec = model()
    unknown = [s for s in req.liked + req.disliked if s not in rec.id_to_idx]
    if unknown:
        raise HTTPException(404, f"Unknown show_id(s): {unknown}")
    out = rec.profile(req.liked, req.disliked, k=req.k,
                      filters=Filters(content_type=req.content_type), diversity=req.diversity)
    return records(out, ["score"])


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50)):
    return records(model().search(q, k=k), ["score"])


@app.get("/stats")
def stats():
    df = model().df
    return {
        "titles": len(df),
        "by_type": df["type"].value_counts().to_dict(),
        "top_genres": df["genres"].str.split("|").explode().value_counts().head(10).to_dict(),
        "top_countries": df["primary_country"].value_counts().head(10).to_dict(),
        "release_year_range": [int(df["release_year"].min()), int(df["release_year"].max())],
    }


@app.get("/users/{username}/watchlist")
def get_watchlist(username: str):
    return {"username": username, "show_ids": db.get_watchlist(username)}


@app.post("/users/{username}/watchlist", status_code=201)
def add_watchlist(username: str, item: WatchItem):
    if item.show_id not in model().id_to_idx:
        raise HTTPException(404, "Title not found")
    db.add_to_watchlist(username, item.show_id)
    return {"ok": True}


@app.delete("/users/{username}/watchlist/{show_id}")
def delete_watchlist(username: str, show_id: str):
    db.remove_from_watchlist(username, show_id)
    return {"ok": True}
