import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import FEATURE_WEIGHTS
from src.ui import get_clusters, get_evaluation, get_recommender, get_titles, title_picker

df = get_titles()
rec = get_recommender()
st.title("🧠 ML Model Insights")

st.markdown("""
### Approach: hybrid content-based filtering
The Kaggle dataset contains **no user ratings**, so collaborative filtering is not possible.
Instead every title is turned into a weighted, multi-field **TF-IDF vector** and compared with
**cosine similarity**:
""")
st.latex(r"\text{sim}(a,b)=\sum_{f\in F} w_f\cdot\cos\!\big(\mathbf{v}^{(f)}_a,\mathbf{v}^{(f)}_b\big),\qquad \sum_f w_f = 1")
c1, c2 = st.columns([1, 1.4])
with c1:
    w = pd.DataFrame({"Feature": list(FEATURE_WEIGHTS), "Weight": list(FEATURE_WEIGHTS.values())})
    fig = px.bar(w, x="Weight", y="Feature", orientation="h", text="Weight", title="Feature weights")
    fig.update_traces(marker_color="#E50914", texttemplate="%{text:.2f}", textposition="outside",
                      cliponaxis=False)
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title=None, yaxis_title=None,
                      xaxis_showticklabels=False, xaxis_showgrid=False, height=300)
    st.plotly_chart(fig, width="stretch")
with c2:
    vocab = {n: len(v.vocabulary_) for n, v in rec.vectorizers.items()}
    st.markdown(f"""
| Pipeline step | Detail |
|---|---|
| Data cleaning | fixed shifted `duration` values, parsed dates, removed duplicates, filled missing |
| Feature engineering | 33 columns: runtime, seasons, maturity group, decade, lag to Netflix… |
| Description | TF-IDF, 1–2-grams, stop-words removed, sublinear TF → **{vocab['description']:,}** terms |
| Genres / Cast / Director / Country | one token per entity → **{vocab['genres']}** / **{vocab['cast']:,}** / **{vocab['director']:,}** / **{vocab['country']}** |
| Final matrix | **{rec.X.shape[0]:,} × {rec.X.shape[1]:,}** sparse, {rec.X.nnz:,} non-zeros |
| Taste profile | Rocchio: mean(liked) − 0.5·mean(disliked) |
| Diversity | Maximal Marginal Relevance re-ranking |
| Clustering | LSA (TruncatedSVD 100) → K-Means (k=12) → t-SNE map |
""")

# -------------------------------------------------------------- evaluation
st.divider()
st.subheader("📏 Offline evaluation vs baselines")
ev = get_evaluation()
if ev is not None:
    st.caption("400 random query titles, top-10 recommendations each. Proxy metrics, because "
               "the dataset has no ratings. Higher is better for all except latency.")
    fmt = {c: "{:.1%}" for c in ev.columns if "@10" in c or c in ("Intra-list Diversity", "Catalog Coverage")}
    fmt.update({"Novelty": "{:.2f}", "Latency (ms/query)": "{:.1f}"})
    st.dataframe(ev.style.format(fmt).highlight_max(
        subset=["Genre Precision@10", "People Hit@10", "Plot Similarity@10"], color="#4a0a0e"),
        hide_index=True, width="stretch")
    metrics = ["Genre Precision@10", "Genre Jaccard@10", "People Hit@10", "Plot Similarity@10"]
    sel = st.radio("Compare metric", metrics, horizontal=True)
    fig = px.bar(ev, x=sel, y="Method", orientation="h", text=sel, title=sel)
    fig.update_traces(marker_color=["#5c5c5c", "#5c5c5c", "#5c5c5c", "#E50914", "#E50914"],
                      texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title=None, yaxis_title=None,
                      xaxis_showticklabels=False, xaxis_showgrid=False, height=320)
    st.plotly_chart(fig, width="stretch")
    h = ev.set_index("Method")
    hy, go_, do = h.loc["Hybrid weighted (ours)"], h.loc["Genre-only"], h.loc["Description-only (TF-IDF)"]
    st.success(
        f"**Key result:** the hybrid model keeps genre precision at **{hy['Genre Precision@10']:.1%}** "
        f"(vs {do['Genre Precision@10']:.1%} for description-only) while finding titles that share "
        f"a director or actor **{hy['People Hit@10'] / go_['People Hit@10']:.1f}×** more often and with "
        f"**{hy['Plot Similarity@10'] / go_['Plot Similarity@10']:.1f}×** the plot similarity of the "
        f"genre-only baseline. It balances every signal instead of maximising one.")
else:
    st.info("Run `python -m src.pipeline` to generate the evaluation report.")

# -------------------------------------------------------------- clusters
st.divider()
st.subheader("🗺️ Content map: 12 clusters discovered by K-Means")
cl = get_clusters()
if cl is not None and "tsne_x" in df:
    sil = cl["silhouette"].iloc[0]
    st.caption(f"Each dot is a title, positioned by t-SNE on its LSA embedding. Silhouette score "
               f"(cosine): **{sil:.3f}**. Use the selector to highlight one cluster.")
    choice = st.selectbox("Highlight cluster", ["All"] + [f"{r.cluster}: {r.cluster_name}" for r in cl.itertuples()])
    plot = df[["title", "type", "release_year", "tsne_x", "tsne_y", "cluster", "cluster_name"]].copy()
    if choice == "All":
        fig = px.scatter(plot, x="tsne_x", y="tsne_y", color="cluster_name", hover_name="title",
                         hover_data={"type": True, "release_year": True, "tsne_x": False,
                                     "tsne_y": False, "cluster_name": False},
                         color_discrete_sequence=px.colors.qualitative.Dark24, height=620)
        fig.update_traces(marker_size=4, marker_opacity=0.75)
        # direct labels at centroids so identity never relies on colour alone
        for name, grp in plot.groupby("cluster_name"):
            fig.add_annotation(x=grp["tsne_x"].median(), y=grp["tsne_y"].median(), text=f"<b>{name}</b>",
                               showarrow=False, font=dict(color="#ffffff", size=11),
                               bgcolor="rgba(20,20,20,.75)", borderpad=3)
    else:
        cid = int(choice.split(":")[0])
        plot["Group"] = (plot["cluster"] == cid).map({True: choice.split(": ", 1)[1], False: "Other"})
        fig = px.scatter(plot.sort_values("Group", key=lambda s: s != "Other"), x="tsne_x", y="tsne_y",
                         color="Group", hover_name="title",
                         color_discrete_map={"Other": "#3a3a3a", plot.loc[plot["cluster"] == cid, "Group"].iloc[0]: "#E50914"},
                         height=620)
        fig.update_traces(marker_size=5)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title=None,
                      legend=dict(orientation="v", x=1.01, y=1))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cl.drop(columns=["silhouette"]).style.format({"pct_movies": "{:.0%}"}),
                 hide_index=True, width="stretch")

# -------------------------------------------------------------- pairwise explorer
st.divider()
st.subheader("🔬 Similarity explorer")
a, b = st.columns(2)
with a:
    s1 = title_picker("Title A", key="pa", default="Narcos (2017, TV Show)")
with b:
    s2 = title_picker("Title B", key="pb", default="El Chapo (2018, TV Show)")
if s1 and s2 and s1 != s2:
    i, j = rec.idx(s1), rec.idx(s2)
    rows = []
    for n, B in rec.blocks.items():
        cos = float((B[i] @ B[j].T).toarray()[0, 0])
        rows.append({"Feature": n, "Cosine": cos, "Weight": rec.weights[n], "Contribution": cos * rec.weights[n]})
    t = pd.DataFrame(rows)
    st.metric("Overall similarity", f"{t['Contribution'].sum():.1%}")
    st.dataframe(t.style.format({"Cosine": "{:.3f}", "Weight": "{:.2f}", "Contribution": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.markdown(f"<div class='nf-reason'>{rec.explain(s1, s2)}</div>", unsafe_allow_html=True)
