import html

import pandas as pd
import plotly.express as px
import streamlit as st

from src.classifier import AUDIENCES
from src.ui import RED, get_classifier, get_recommender, render_grid

clf = get_classifier()
rec = get_recommender()
rep = clf.report

st.title("🔮 AI Genre Predictor")
st.caption("Supervised NLP: write any plot synopsis (a real one or your own idea) and three trained "
           "models predict its **genres**, whether it is a **movie or a series**, and its "
           "**target audience**, then explain which words drove each prediction.")

EXAMPLES = {
    "🚀 Kids space adventure": ("Galaxy Kids", "A group of young friends build a rocket in their backyard "
                                "and blast off on a funny adventure across the solar system."),
    "🔪 Crime thriller": ("Blood Money", "A detective goes undercover inside a ruthless drug cartel to "
                          "find the killer of her partner, but the deeper she goes the more corrupt everyone looks."),
    "💕 Rom-com": ("Second Chances", "Two rival wedding planners are forced to work together and slowly "
                  "fall in love while organising the wedding of the year."),
    "🌍 Nature docuseries": ("Wild Earth", "This docuseries explores the planet's most remote habitats and "
                            "the extraordinary animals that survive there."),
    "🎤 Stand-up": ("Unfiltered", "The comedian riffs on marriage, parenting and getting older in this "
                   "hilarious stand-up special recorded live."),
}
cols = st.columns(len(EXAMPLES))
for col, (label, (t, d)) in zip(cols, EXAMPLES.items()):
    if col.button(label, width="stretch"):
        st.session_state["pred_title"], st.session_state["pred_desc"] = t, d

st.session_state.setdefault("pred_title", EXAMPLES["🔪 Crime thriller"][0])
st.session_state.setdefault("pred_desc", EXAMPLES["🔪 Crime thriller"][1])
title = st.text_input("Title", key="pred_title")
desc = st.text_area("Plot synopsis", key="pred_desc", height=100)

if desc.strip():
    out = clf.predict(title, desc)
    if out["known_terms"] == 0:
        st.warning("None of these words appear in the model's vocabulary. Try a longer English description.")
        st.stop()

    g = out["genres"]
    chosen = g[g["Predicted"]]
    st.markdown("#### Predicted genres")
    st.markdown("".join(
        f'<span class="nf-pill" style="border-color:#E50914;font-size:.95rem">{html.escape(r.Genre)} · '
        f'{r.Probability:.0%}</span>' for r in chosen.itertuples()), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        top = g.head(10).iloc[::-1]
        fig = px.bar(top, x="Probability", y="Genre", orientation="h", text="Probability",
                     title="Top-10 genre probabilities", hover_data={"Key words": True})
        fig.update_traces(marker_color=[RED if p else "#5c5c5c" for p in top["Predicted"]],
                          texttemplate="%{text:.0%}", textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_range=[0, 1.15], xaxis_tickformat=".0%", xaxis_title=None,
                          yaxis_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Red = above the genre's tuned decision threshold. Hover to see the key words.")
    with c2:
        t = pd.Series(out["type"]).reindex(["Movie", "TV Show"])
        st.markdown("#### Format")
        st.metric("Prediction", t.idxmax(), f"{t.max():.0%} confidence", delta_color="off")
        fig = px.bar(x=t.index, y=t.values, text=t.values, color=t.index,
                     color_discrete_map={"Movie": RED, "TV Show": "#3987e5"})
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.15], yaxis_visible=False,
                          xaxis_title=None, height=240)
        st.plotly_chart(fig, width="stretch")
    with c3:
        a = pd.Series(out["audience"]).reindex(AUDIENCES)
        st.markdown("#### Audience")
        st.metric("Prediction", a.idxmax(), f"{a.max():.0%} confidence", delta_color="off")
        fig = px.bar(x=a.index, y=a.values, text=a.values)
        fig.update_traces(marker_color=RED, texttemplate="%{text:.0%}", textposition="outside",
                          cliponaxis=False)
        fig.update_layout(yaxis_range=[0, 1.15], yaxis_visible=False, xaxis_title=None, height=240)
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🧩 Why? Words that pushed each predicted genre up")
    for r in chosen.itertuples():
        st.markdown(f"<div class='nf-reason'><b>{html.escape(r.Genre)}</b>: "
                    f"{html.escape(r._4) or '(prior / overall style)'}</div>", unsafe_allow_html=True)

    st.markdown("#### 🎬 Existing Netflix titles closest to this idea")
    render_grid(rec.search(f"{title} {desc}", k=6), key="pred-sim", cols=6, show_score=False)

# ---------------------------------------------------------------- evaluation
st.divider()
st.subheader("📏 How good are these models? (held-out test set)")
st.caption(f"Random split: {rep.n_train:,} training titles, a validation set for tuning "
           f"per-genre thresholds, and {rep.n_test:,} unseen test titles. Features: TF-IDF of title + "
           "one-sentence synopsis only (no cast, country or other metadata).")
m = st.columns(4)
m[0].metric("Genre micro-F1", f"{rep.genre_micro_f1:.1%}",
            f"+{rep.genre_micro_f1 - rep.genre_baseline_micro_f1:.1%} vs baseline")
m[1].metric("Genre macro-F1", f"{rep.genre_macro_f1:.1%}")
m[2].metric("Movie vs TV accuracy", f"{rep.type_accuracy:.1%}",
            f"+{rep.type_accuracy - rep.type_baseline:.1%} vs majority class")
m[3].metric("Audience accuracy", f"{rep.audience_accuracy:.1%}",
            f"+{rep.audience_accuracy - rep.audience_baseline:.1%} vs majority class")

c1, c2 = st.columns([1.3, 1])
with c1:
    pg = rep.per_genre.sort_values("F1")
    fig = px.bar(pg, x="F1", y="Genre", orientation="h", title="F1 score per genre (test set)",
                 hover_data={"Precision": ":.2f", "Recall": ":.2f", "Test support": True})
    fig.update_traces(marker_color=RED)
    fig.update_layout(height=900, xaxis_range=[0, 1], xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
with c2:
    for cm, labels, name in [(rep.type_cm, ["Movie", "TV Show"], "Movie vs TV Show"),
                             (rep.audience_cm, AUDIENCES, "Audience")]:
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True, title=f"Confusion matrix: {name}",
                        color_continuous_scale=["#1f1f1f", "#7a0a10", "#E50914", "#ffc2c5"],
                        labels=dict(x="Predicted", y="Actual", color="Titles"))
        fig.update_layout(height=420 if len(labels) > 2 else 330, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

st.info("**Interpretation.** Genres with distinctive vocabulary (stand-up, documentaries, kids, sports) are "
        "predicted well. Genres defined by *where* a show comes from (Korean, British, Anime) or by format "
        "rather than story are hard to infer from a one-sentence synopsis alone. Adjacent audience levels "
        "(Teens vs Adults) are the most common confusion. All models clearly beat their baselines.")
