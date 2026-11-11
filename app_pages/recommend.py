import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import db
from src.config import FEATURE_WEIGHTS
from src.moods import MOODS, mood_recommend, neighbour_graph
from src.recommender import Filters
from src.ui import RED, get_recommender, get_titles, render_grid, title_picker

df = get_titles()
rec = get_recommender()
user = st.session_state.user

st.title("✨ Get Recommendations")
st.caption("Four ways to discover: item-to-item similarity (with a 2-hop network graph), a "
           "multi-title taste profile, a free-text description, or your current mood.")

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("🎛️ Filters")
    ctype = st.radio("Type", ["Any", "Movie", "TV Show"], horizontal=True)
    yr_min, yr_max = int(df["release_year"].min()), int(df["release_year"].max())
    years = st.slider("Release year", yr_min, yr_max, (1990, yr_max))
    all_genres = sorted(df["genres"].str.split("|").explode().dropna().unique())
    genres = st.multiselect("Genres (any of)", all_genres)
    maturity = st.multiselect("Audience", ["Kids", "Older Kids", "Teens", "Adults", "Unrated"])
    k = st.slider("Number of results", 6, 30, 12, step=6)
    diversity = st.slider("Diversity (MMR)", 0.0, 0.9, 0.0, 0.1,
                          help="0 = most similar first. Higher values re-rank with Maximal Marginal "
                               "Relevance to avoid near-duplicate results.")

filters = Filters(content_type=None if ctype == "Any" else ctype, min_year=years[0],
                  max_year=years[1], genres=genres, maturity=maturity)



def draw_network(show_id: str) -> None:
    """Radial 2-hop similarity graph: centre = chosen title, ring 1 = its top matches,
    ring 2 = their own top matches. Lines = similarity links."""
    nodes, edges = neighbour_graph(rec, show_id)
    pos = {show_id: (0.0, 0.0)}
    ring1 = nodes[nodes["level"] == 1]["show_id"].tolist()
    angle = {}
    for i, s in enumerate(ring1):
        a = 2 * np.pi * i / max(len(ring1), 1)
        angle[s] = a
        pos[s] = (np.cos(a), np.sin(a))
    for parent, grp in nodes[nodes["level"] == 2].groupby("parent"):
        kids = grp["show_id"].tolist()
        for j, s in enumerate(kids):
            a = angle[parent] + (j - (len(kids) - 1) / 2) * 0.22
            pos[s] = (2.1 * np.cos(a), 2.1 * np.sin(a))
    ex, ey = [], []
    for a, b, _ in edges:
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                               line=dict(color="#3a3a3a", width=1), showlegend=False))
    styles = {0: ("Selected title", RED, 26), 1: ("Direct matches", "#3987e5", 16),
              2: ("2nd-hop matches", "#a3a3a3", 10)}
    for lvl, (name, color, size) in styles.items():
        sub = nodes[nodes["level"] == lvl]
        fig.add_trace(go.Scatter(
            x=[pos[s][0] for s in sub["show_id"]], y=[pos[s][1] for s in sub["show_id"]],
            mode="markers+text" if lvl < 2 else "markers", name=name,
            text=sub["title"].str.slice(0, 22), textposition="top center",
            textfont=dict(size=11 if lvl else 14, color="#f5f5f5"),
            marker=dict(size=size, color=color, line=dict(color="#141414", width=2)),
            customdata=np.stack([sub["type"], sub["release_year"], sub["score"]], axis=1),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]} · %{customdata[1]}"
                          "<br>similarity %{customdata[2]:.0%}<extra></extra>"))
    fig.update_layout(height=620, xaxis_visible=False, yaxis_visible=False,
                      yaxis_scaleanchor="x", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(nodes)} titles, {len(edges)} similarity links. Clusters of tightly linked titles "
               "show 'neighbourhoods' of taste; cross-links reveal titles that bridge them.")


tab1, tab2, tab3, tab4 = st.tabs(["🎬 Similar to a title", "🧬 My taste profile", "💬 Describe it",
                                  "🎭 Mood"])

# ----------------------------------------------------------- item-to-item
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1:
        sid = title_picker("Pick a movie or show you like", key="pick-one",
                           default="Stranger Things (2019, TV Show)")
    with c2:
        st.write("")
        custom = st.toggle("Tune model weights", help="Change how much each feature matters.")
    weights = None
    if custom:
        cols = st.columns(len(FEATURE_WEIGHTS))
        weights = {f: cols[i].slider(f.title(), 0.0, 1.0, float(w), 0.05, key=f"w-{f}")
                   for i, (f, w) in enumerate(FEATURE_WEIGHTS.items())}
    if sid:
        out = rec.similar(sid, k=k, filters=filters, diversity=diversity, weights=weights)
        src = df[df["show_id"] == sid].iloc[0]
        st.markdown(f"#### Because you liked **{src['title']}**")
        if out.empty:
            st.warning("No titles match your filters - try relaxing them in the sidebar.")
        else:
            render_grid(out, key="sim", cols=6, ranked=True, show_reason=True)
            with st.expander("🔬 How were these scored? (per-feature contribution)"):
                contrib = out[["title"] + [c for c in out.columns if c.startswith("sim_")]].copy()
                contrib.columns = [c.replace("sim_", "").title() for c in contrib.columns]
                long = contrib.melt(id_vars="Title", var_name="Feature", value_name="Contribution")
                fig = px.bar(long, y="Title", x="Contribution", color="Feature", orientation="h",
                             color_discrete_sequence=["#3987e5", "#d95926", "#199e70", "#c98500",
                                                      "#d55181", "#008300"])
                fig.update_layout(height=40 * len(contrib) + 120, yaxis=dict(autorange="reversed"),
                                  xaxis_title="Weighted cosine similarity (stacked = final score)",
                                  yaxis_title=None)
                fig.update_traces(marker_line_width=2, marker_line_color="#141414")
                st.plotly_chart(fig, width="stretch")
                st.caption("score = Σ wᶠ · cos(titleᶠ, candidateᶠ) over description, genres, cast, "
                           "director, country and meta features.")
            with st.expander("🕸️ Recommendation network: explore 2 hops of similar titles"):
                draw_network(sid)

# ----------------------------------------------------------- taste profile
with tab2:
    st.markdown("Pick several titles you **love** (and optionally some you **dislike**). "
                "We build a Rocchio user profile vector: *mean(liked) − 0.5 · mean(disliked)*.")
    labels = df.set_index("show_id")["label"]
    watch = db.get_watchlist(user)
    fb = db.get_feedback(user)
    default_liked = list(dict.fromkeys(watch + [s for s, v in fb.items() if v]))[:10]
    liked = st.multiselect("Titles I love", labels.index, default=default_liked,
                           format_func=lambda s: labels[s], key="liked")
    disliked = st.multiselect("Titles I don't enjoy (optional)", labels.index,
                              default=[s for s, v in fb.items() if not v],
                              format_func=lambda s: labels[s], key="disliked")
    if liked:
        out = rec.profile(liked, disliked, k=k, filters=filters, diversity=diversity)
        if out.empty:
            st.warning("No titles match your filters.")
        else:
            render_grid(out, key="prof", cols=6, ranked=True)
    else:
        st.info("Add at least one title you love. Tip: titles in **My List** and 👍 likes are pre-filled.")

# ----------------------------------------------------------- free text
with tab3:
    text = st.text_area("Describe the kind of content you want",
                        "A gripping true crime documentary about a famous murder case",
                        height=90)
    if text.strip():
        out = rec.search(text, k=k, filters=filters, diversity=diversity)
        if out.empty:
            st.warning("Nothing matched - try other keywords or relax filters.")
        else:
            render_grid(out, key="txt", cols=6, ranked=True)

# ----------------------------------------------------------- mood
with tab4:
    st.markdown("How are you feeling tonight? Each mood combines a keyword profile with genre "
                "affinity: *0.55 · keyword relevance + 0.45 · genre fit*.")
    mood = st.pills("Mood", list(MOODS), default=list(MOODS)[0], label_visibility="collapsed")
    if mood:
        out = mood_recommend(rec, mood, k=k, filters=filters, diversity=max(diversity, 0.3))
        if out.empty:
            st.warning("No titles match your filters.")
        else:
            render_grid(out, key="mood", cols=6, ranked=True, show_score=False)
