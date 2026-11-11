import streamlit as st

from src import db
from src.ui import get_recommender, get_titles, render_grid

df = get_titles()
rec = get_recommender()
user = st.session_state.user

st.title("❤️ My List")
st.caption(f"Saved for profile **{user}** (SQLite database). Change the profile name in the sidebar.")

ids = db.get_watchlist(user)
if not ids:
    st.info("Your list is empty. Use the **＋** button on any title card to save it here.")
    st.stop()

saved = df.set_index("show_id").loc[ids].reset_index()
render_grid(saved, key="wl", cols=6, show_score=False)

st.divider()
st.subheader("🎯 Recommended because of your list")
fb = db.get_feedback(user)
disliked = [s for s, v in fb.items() if not v]
diversity = st.slider("Diversity", 0.0, 0.9, 0.3, 0.1)
recs = rec.profile(ids, disliked, k=12, diversity=diversity)
render_grid(recs, key="wl-rec", cols=6, ranked=True)

c = st.columns(3)
c[0].metric("Saved titles", len(ids))
c[1].metric("Movies / Shows", f"{(saved['type'] == 'Movie').sum()} / {(saved['type'] == 'TV Show').sum()}")
c[2].metric("Top genre", saved["genres"].str.split("|").explode().value_counts().index[0])
