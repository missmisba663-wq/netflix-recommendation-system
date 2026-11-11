import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import db
from src.data_access import build_recommender, load_titles
from src.recommender import Filters


@pytest.fixture(scope="session")
def rec():
    return build_recommender(load_titles())


def sid(rec, title):
    return rec.df.loc[rec.df["title"] == title, "show_id"].iloc[0]


def test_data_is_clean(rec):
    df = rec.df
    assert len(df) > 8000
    assert df["show_id"].is_unique
    assert not df["rating"].str.contains("min").any()          # shifted values fixed
    assert df.loc[df["type"] == "Movie", "duration_min"].notna().mean() > 0.99


def test_similar_excludes_self_and_is_sorted(rec):
    s = sid(rec, "Stranger Things")
    out = rec.similar(s, k=10)
    assert len(out) == 10 and s not in set(out["show_id"])
    assert out["score"].is_monotonic_decreasing
    # contributions add up to the final score
    contrib = out[[c for c in out.columns if c.startswith("sim_")]].sum(axis=1)
    assert (abs(contrib - out["score"]) < 1e-4).all()


def test_filters_are_respected(rec):
    out = rec.similar(sid(rec, "Narcos"), k=12, filters=Filters(content_type="Movie", min_year=2015))
    assert (out["type"] == "Movie").all() and (out["release_year"] >= 2015).all()


def test_custom_weights_do_not_mutate_model(rec):
    before = dict(rec.weights)
    rec.similar(sid(rec, "Narcos"), k=5, weights={"genres": 1})
    assert rec.weights == before


def test_profile_and_search(rec):
    liked = [sid(rec, "Narcos"), sid(rec, "Ozark")]
    out = rec.profile(liked, k=8, diversity=0.3)
    assert len(out) == 8 and not set(liked) & set(out["show_id"])
    res = rec.search("korean zombie", k=5)
    assert len(res) > 0


def test_api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(db.connect, "__defaults__", (tmp_path / "t.db",))
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    hit = c.get("/titles", params={"q": "narcos"}).json()[0]
    r = c.get(f"/recommend/{hit['show_id']}", params={"k": 5}).json()
    assert len(r["recommendations"]) == 5
    assert c.get("/recommend/nope").status_code == 404
    assert c.post("/recommend/profile", json={"liked": [hit["show_id"]], "k": 3}).status_code == 200
    assert c.post("/users/ann/watchlist", json={"show_id": hit["show_id"]}).status_code == 201
    assert c.get("/users/ann/watchlist").json()["show_ids"] == [hit["show_id"]]
