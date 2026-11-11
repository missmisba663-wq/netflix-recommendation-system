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


def test_moods_and_graph(rec):
    from src.moods import MOODS, mood_recommend, neighbour_graph
    out = mood_recommend(rec, "👻 Scary", k=10)
    assert len(out) == 10
    assert out["genres"].str.contains("Horror").mean() >= 0.7
    nodes, edges = neighbour_graph(rec, sid(rec, "Narcos"))
    assert nodes["show_id"].is_unique and len(edges) >= len(nodes) - 1
    assert len(MOODS) >= 8


def test_classifier(rec):
    from src.classifier import ContentClassifier
    clf = ContentClassifier().fit(rec.df)
    r = clf.report
    assert r.genre_micro_f1 > r.genre_baseline_micro_f1 + 0.1
    assert r.type_accuracy > r.type_baseline
    assert r.audience_accuracy > r.audience_baseline
    p = clf.predict("Funny Bones", "The comedian riffs on dating and family in this hilarious stand-up special.")
    assert p["genres"].iloc[0]["Genre"] == "Stand-Up Comedy"


def test_poster_matching_rules():
    from src.posters import _candidates, _desc_ok, _norm
    assert _norm("Narcos: Mexico (2018 TV series)") == _norm("Narcos: Mexico")
    assert _candidates("Mute", 2018, "Movie")[0][0] == "Mute (2018 film)"
    assert _desc_ok("2018 film by Duncan Jones", 2018, "Movie")
    assert not _desc_ok("1999 film by Dennis Dugan", 2018, "Movie")   # same name, wrong year
    assert not _desc_ok("Disambiguation page", 2018, "Movie")
