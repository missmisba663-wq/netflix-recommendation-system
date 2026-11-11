"""Shared Streamlit UI: cached resources, theme, title cards, details dialog."""
from __future__ import annotations

import hashlib
import html
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st

from . import db
from .config import CLUSTER_CSV, EVAL_CSV, LIST_SEP
from .data_access import build_recommender, load_titles

RED = "#E50914"
BLUE = "#3987e5"
TYPE_COLORS = {"Movie": RED, "TV Show": BLUE}
INK = "#e5e5e5"
MUTED = "#a3a3a3"
GRID = "#2a2a2a"


# ----------------------------------------------------------------- resources
@st.cache_data(show_spinner="Loading Netflix catalogue...")
def get_titles() -> pd.DataFrame:
    return load_titles()


@st.cache_resource(show_spinner="Training recommendation model...")
def get_recommender():
    return build_recommender(get_titles())


@st.cache_resource(show_spinner="Training NLP classifiers...")
def get_classifier():
    from .classifier import ContentClassifier
    return ContentClassifier().fit(get_titles())


@st.cache_data
def get_evaluation() -> pd.DataFrame | None:
    return pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None


@st.cache_data
def get_clusters() -> pd.DataFrame | None:
    return pd.read_csv(CLUSTER_CSV) if CLUSTER_CSV.exists() else None


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.environ.get(name, default)))
    except Exception:  # no secrets.toml
        return os.environ.get(name, default)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def poster_url(title: str, year: int, kind: str) -> str | None:
    """Optional real posters via TMDB (set TMDB_API_KEY in secrets)."""
    key = get_secret("TMDB_API_KEY")
    if not key:
        return None
    endpoint = "movie" if kind == "Movie" else "tv"
    year_param = "year" if kind == "Movie" else "first_air_date_year"
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/search/{endpoint}",
            params={"api_key": key, "query": title, year_param: year},
            timeout=4,
        )
        results = r.json().get("results") or []
        path = next((x.get("poster_path") for x in results if x.get("poster_path")), None)
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


# --------------------------------------------------------------------- theme
def plotly_theme() -> None:
    tmpl = go.layout.Template(pio.templates["plotly_dark"])
    tmpl.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK, size=13),
        colorway=[RED, BLUE, "#199e70", "#c98500", "#d55181", "#9085e9"],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        hoverlabel=dict(bgcolor="#1f1f1f", bordercolor="#444", font_color=INK),
        margin=dict(l=10, r=10, t=56, b=10),
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_color=MUTED),
    )
    pio.templates["netflix"] = tmpl
    pio.templates.default = "netflix"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.nf-logo { font-family: 'Bebas Neue', sans-serif; color: #E50914; font-size: 2.4rem;
           letter-spacing: .06em; line-height: 1; margin: 0; }
.nf-hero { position: relative; border-radius: 14px; padding: 48px 40px; margin-bottom: 26px;
           background: radial-gradient(ellipse at 80% 20%, rgba(229,9,20,.55), transparent 55%),
                       linear-gradient(110deg, #000 0%, #1a0003 55%, #3a0006 100%);
           border: 1px solid #2a2a2a; overflow: hidden; }
.nf-hero h1 { font-family: 'Bebas Neue', sans-serif; font-size: 3.6rem; margin: 0; line-height: 1;
              letter-spacing: .03em; color: #fff; }
.nf-hero p { color: #d4d4d4; font-size: 1.05rem; max-width: 720px; margin: 12px 0 0; }
.nf-hero .tag { display:inline-block; background:#E50914; color:#fff; padding:3px 10px; border-radius:4px;
                font-weight:700; font-size:.75rem; letter-spacing:.08em; margin-bottom:12px; }
.nf-kpi { background:#181818; border:1px solid #2a2a2a; border-radius:12px; padding:16px 18px; }
.nf-kpi .v { font-size:1.9rem; font-weight:800; color:#fff; line-height:1.1; }
.nf-kpi .l { color:#a3a3a3; font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; }
.nf-row-title { font-size:1.3rem; font-weight:800; margin: 18px 0 8px; color:#fff; }
.nf-card { border-radius: 8px; overflow: hidden; background:#181818; border:1px solid #262626;
           transition: transform .18s ease, border-color .18s ease; height: 100%; }
.nf-card:hover { transform: translateY(-3px) scale(1.02); border-color:#E50914; }
.nf-poster { position: relative; aspect-ratio: 2/3; display:flex; align-items:flex-end;
             padding: 12px; background-size: cover; background-position: center; }
.nf-img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
.nf-poster .shade.light { background: linear-gradient(180deg, rgba(0,0,0,.35), transparent 25%); }
.nf-name { padding: 9px 11px 0; color:#fff; font-weight:700; font-size:.86rem; white-space:nowrap;
           overflow:hidden; text-overflow:ellipsis; }
.nf-poster .shade { position:absolute; inset:0; background: linear-gradient(180deg, transparent 40%, rgba(0,0,0,.92)); }
.nf-poster .t { position:relative; font-family:'Bebas Neue',sans-serif; font-size:1.55rem; line-height:1.02;
                color:#fff; letter-spacing:.02em; text-shadow:0 2px 8px rgba(0,0,0,.6); word-break: break-word; }
.nf-poster .n { position:absolute; top:10px; left:10px; font-family:'Bebas Neue',sans-serif; color:#E50914;
                font-size:1.4rem; letter-spacing:.05em; }
.nf-badge { position:absolute; top:10px; right:10px; background:rgba(0,0,0,.7); color:#fff; font-size:.65rem;
            font-weight:700; padding:3px 7px; border-radius:4px; letter-spacing:.08em; border:1px solid #444; }
.nf-meta { padding: 9px 11px 4px; color:#d4d4d4; font-size:.78rem; white-space:nowrap; overflow:hidden;
           text-overflow:ellipsis; }
.nf-match { color:#46d369; font-weight:800; }
.nf-genres { padding: 0 11px 10px; color:#8c8c8c; font-size:.72rem; white-space:nowrap; overflow:hidden;
             text-overflow:ellipsis; }
.nf-reason { background:#1f1f1f; border-left:3px solid #E50914; padding:8px 12px; border-radius:4px;
             color:#cfcfcf; font-size:.85rem; margin: 6px 0; }
.nf-pill { display:inline-block; background:#262626; border:1px solid #333; color:#ddd; border-radius:999px;
           padding:2px 10px; margin:2px 4px 2px 0; font-size:.78rem; }
div[data-testid="stButton"] button { border-radius: 6px; }
section[data-testid="stSidebar"] { background:#0b0b0b; border-right:1px solid #1f1f1f; }
footer { visibility: hidden; }
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    plotly_theme()
    if "user" not in st.session_state:
        st.session_state.user = "guest"


def sidebar_profile() -> str:
    with st.sidebar:
        st.markdown('<p class="nf-logo">NETFLIX·REC</p>', unsafe_allow_html=True)
        st.caption("AI-powered movie & TV recommender")
        name = st.text_input("👤 Profile name", value=st.session_state.user, max_chars=30,
                             help="Your watchlist and likes are saved under this name.")
        st.session_state.user = (name or "guest").strip().lower()
        n = len(db.get_watchlist(st.session_state.user))
        st.caption(f"My List: **{n}** titles")
    return st.session_state.user


# --------------------------------------------------------------------- cards
def _hue(title: str) -> int:
    return int(hashlib.md5(title.encode()).hexdigest()[:6], 16) % 360


def fmt_duration(row) -> str:
    if row["type"] == "Movie" and pd.notna(row.get("duration_min")):
        m = int(row["duration_min"])
        return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"
    if pd.notna(row.get("seasons")):
        s = int(row["seasons"])
        return f"{s} Season{'s' if s > 1 else ''}"
    return ""


@st.cache_data(show_spinner=False)
def _posters_cached(mtime: float) -> dict[str, str]:
    from .posters import load_posters
    return load_posters()


def get_posters() -> dict[str, str]:
    from .posters import POSTERS_CSV
    # cache key = file modification time, so a rebuilt posters.csv is picked up automatically
    return _posters_cached(POSTERS_CSV.stat().st_mtime if POSTERS_CSV.exists() else 0.0)


def thumbnail(row) -> str | None:
    """Wikipedia poster (pre-computed), else TMDB if a key is configured."""
    url = get_posters().get(row["show_id"]) or poster_url(
        str(row["title"]), int(row["release_year"]), row["type"])
    return url.replace("'", "%27").replace('"', "%22") if url else None


def card_html(row, rank: int | None = None, show_score: bool = True) -> str:
    title = html.escape(str(row["title"]))
    img = thumbnail(row)
    h = _hue(str(row["title"]))
    bg = (f"background: radial-gradient(circle at 30% 20%, hsl({h},70%,32%), transparent 60%),"
          f" linear-gradient(160deg, hsl({(h + 40) % 360},55%,18%), #050505)")
    img_tag = f'<img class="nf-img" src="{img}" alt="" loading="lazy">' if img else ""
    match = ""
    if show_score and "score" in row and pd.notna(row["score"]):
        match = f'<div class="nf-match">{round(float(row["score"]) * 100)}% match</div>'
    genres = html.escape(str(row["genres"]).replace(LIST_SEP, " • "))
    num = f'<div class="n">#{rank}</div>' if rank else ""
    badge = "SERIES" if row["type"] == "TV Show" else "FILM"
    return f"""
    <div class="nf-card">
      <div class="nf-poster" style="{bg}">
        {img_tag}<div class="shade{' light' if img else ''}"></div>{num}<span class="nf-badge">{badge}</span>
        <div class="t">{title if not img else ''}</div>
      </div>
      {f'<div class="nf-name">{title}</div>' if img else ''}
      <div class="nf-meta">{match}{int(row['release_year'])} · {html.escape(str(row['rating']))} · {fmt_duration(row)}</div>
      <div class="nf-genres">{genres}</div>
    </div>"""


def render_grid(df: pd.DataFrame, key: str, cols: int = 6, show_score: bool = True,
                ranked: bool = False, show_reason: bool = False) -> None:
    user = st.session_state.get("user", "guest")
    watch = set(db.get_watchlist(user))
    rows = list(df.iterrows())
    for start in range(0, len(rows), cols):
        columns = st.columns(cols)
        for j, (col, (_, row)) in enumerate(zip(columns, rows[start:start + cols])):
            with col:
                st.markdown(card_html(row, rank=start + j + 1 if ranked else None,
                                      show_score=show_score), unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                sid = row["show_id"]
                if b1.button("ⓘ", key=f"{key}-d-{sid}", help="Details & similar titles",
                             width="stretch"):
                    details_dialog(sid)
                in_list = sid in watch
                if b2.button("✓" if in_list else "＋", key=f"{key}-w-{sid}",
                             help="Remove from My List" if in_list else "Add to My List",
                             width="stretch"):
                    (db.remove_from_watchlist if in_list else db.add_to_watchlist)(user, sid)
                    st.rerun()
                if show_reason and "reason" in row:
                    with st.popover("Why?", width="stretch"):
                        st.write(row["reason"])


@st.dialog("Title details", width="large")
def details_dialog(show_id: str) -> None:
    rec = get_recommender()
    row = rec.df.iloc[rec.idx(show_id)]
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown(card_html(row, show_score=False), unsafe_allow_html=True)
    with c2:
        st.markdown(f"## {row['title']}")
        st.markdown(f"**{row['type']}** · {int(row['release_year'])} · {row['rating']} · "
                    f"{fmt_duration(row)} · _{row['maturity_group']}_")
        st.write(row["description"])
        def pills(label, value):
            items = [x for x in str(value).split(LIST_SEP) if x]
            if items:
                st.markdown(f"**{label}:** " + "".join(
                    f'<span class="nf-pill">{html.escape(i)}</span>' for i in items[:8]),
                    unsafe_allow_html=True)
        pills("Genres", row["genres"])
        pills("Director", row["directors"])
        pills("Cast", row["cast_list"])
        pills("Country", row["countries"])
        if pd.notna(row.get("date_added")) and str(row.get("date_added")):
            st.caption(f"Added to Netflix: {pd.to_datetime(row['date_added']):%d %B %Y} · "
                       f"Content cluster: {row.get('cluster_name', '–')}")
        user = st.session_state.get("user", "guest")
        fb = db.get_feedback(user).get(show_id)
        a, b, c = st.columns(3)
        if a.button("👍 Like" + (" ✓" if fb is True else ""), key=f"dl-{show_id}"):
            db.set_feedback(user, show_id, True); st.rerun()
        if b.button("👎 Not for me" + (" ✓" if fb is False else ""), key=f"dd-{show_id}"):
            db.set_feedback(user, show_id, False); st.rerun()
        in_list = show_id in db.get_watchlist(user)
        if c.button("✓ In My List" if in_list else "＋ My List", key=f"dw-{show_id}"):
            (db.remove_from_watchlist if in_list else db.add_to_watchlist)(user, show_id)
            st.rerun()
    st.markdown("#### More like this")
    sims = rec.similar(show_id, k=6)
    for _, r in sims.iterrows():
        st.markdown(f"**{r['title']}** ({int(r['release_year'])}, {r['type']}) — "
                    f"<span class='nf-match'>{r['score']:.0%} similarity</span>"
                    f"<div class='nf-reason'>{html.escape(r['reason'])}</div>",
                    unsafe_allow_html=True)


def kpi(col, value, label) -> None:
    col.markdown(f'<div class="nf-kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def title_picker(label: str, key: str, default: str | None = None) -> str | None:
    """Searchable selectbox over all titles; returns show_id."""
    df = get_titles()
    labels = df["label"].tolist()
    ids = df["show_id"].tolist()
    idx = labels.index(default) if default in labels else None
    choice = st.selectbox(label, options=range(len(labels)), index=idx, key=key,
                          format_func=lambda i: labels[i], placeholder="Start typing a title...")
    return ids[choice] if choice is not None else None
