import html

import streamlit as st

from src import db
from src.ui import get_recommender, get_titles, kpi, render_grid

df = get_titles()
rec = get_recommender()
user = st.session_state.user

st.markdown("""
<div class="nf-hero">
  <span class="tag">AI · MACHINE LEARNING</span>
  <h1>Find your next favourite show</h1>
  <p>A hybrid content-based recommender trained on the real Netflix catalogue of 8,800+ movies and
  TV shows from Kaggle. Pick a title you love, describe your mood, or build a taste profile and our
  model finds the best matches and explains why.</p>
</div>
""", unsafe_allow_html=True)

c = st.columns(5)
kpi(c[0], f"{len(df):,}", "Titles")
kpi(c[1], f"{(df['type'] == 'Movie').sum():,}", "Movies")
kpi(c[2], f"{(df['type'] == 'TV Show').sum():,}", "TV Shows")
kpi(c[3], f"{df['countries'].str.split('|').explode().replace('', None).nunique():,}", "Countries")
kpi(c[4], f"{df['genres'].str.split('|').explode().nunique()}", "Genres")

st.write("")
q = st.text_input("🔎 Describe what you feel like watching",
                  placeholder="e.g. 'dark korean crime thriller', 'feel-good cooking show', 'space adventure for kids'")
if q:
    res = rec.search(q, k=12)
    st.markdown(f'<div class="nf-row-title">Results for "{html.escape(q)}"</div>', unsafe_allow_html=True)
    if res.empty:
        st.info("No matches. Try different words.")
    else:
        render_grid(res, key="home-search", cols=6, show_score=False)

watch = db.get_watchlist(user)
if watch:
    recs = rec.profile(watch, k=6)
    st.markdown('<div class="nf-row-title">🎯 Top picks for you (based on My List)</div>',
                unsafe_allow_html=True)
    render_grid(recs, key="home-foryou", cols=6, ranked=True)

st.markdown('<div class="nf-row-title">🆕 Recently added to Netflix</div>', unsafe_allow_html=True)
recent = df.dropna(subset=["date_added"]).sort_values("date_added", ascending=False).head(6)
render_grid(recent, key="home-recent", cols=6, show_score=False)

st.markdown('<div class="nf-row-title">🎲 Explore a genre</div>', unsafe_allow_html=True)
genres = sorted(df["genres"].str.split("|").explode().dropna().unique())
g = st.pills("Genre", genres, default="International TV Shows" if "International TV Shows" in genres else genres[0],
             label_visibility="collapsed")
if g:
    sub = df[df["genres"].str.contains(g, regex=False)].sort_values("release_year", ascending=False)
    st.session_state.setdefault("genre_seed", 7)
    if st.button("🔀 Shuffle"):
        st.session_state.genre_seed += 1
    render_grid(sub.sample(min(6, len(sub)), random_state=st.session_state.genre_seed),
                key="home-genre", cols=6, show_score=False)
