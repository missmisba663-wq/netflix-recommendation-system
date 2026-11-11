import streamlit as st

st.title("ℹ️ About the Project")
st.markdown("""
**Netflix Movie & TV Show Recommendation System**: an end-to-end machine learning and full-stack
data science project.

#### 🎯 Objectives
1. Clean and analyse the real Netflix catalogue (Kaggle, 8,800+ titles).
2. Build an explainable recommendation engine using NLP and machine learning.
3. Deliver it through a responsive web application with user profiles and a persistent watchlist.
4. Provide a REST API for integration with other clients.
5. Build an interactive Power BI business-intelligence dashboard.
6. Deploy the application live on Streamlit Community Cloud.

#### 📦 Dataset
[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) by Shivam Bansal
(Kaggle). 12 columns: `show_id, type, title, director, cast, country, date_added,
release_year, rating, duration, listed_in, description`.
""")

st.markdown("#### 🏗️ System architecture")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1f1f1f", color="#E50914", fontcolor="white",
        fontname="Helvetica", fontsize=11];
  edge [color="#888888"];
  kaggle [label="Kaggle dataset\\nnetflix_titles.csv"];
  prep [label="Data cleaning &\\nfeature engineering\\n(pandas)"];
  model [label="Hybrid TF-IDF\\nrecommender\\n(scikit-learn)"];
  clus [label="LSA + K-Means\\n+ t-SNE"];
  clf [label="Supervised NLP\\ngenre / type / audience"];
  eval [label="Offline evaluation\\nvs baselines"];
  st [label="Streamlit web app\\n(frontend + UI)"];
  api [label="FastAPI REST API"];
  db [label="SQLite\\nwatchlist & likes"];
  pbi [label="Power BI\\nstar schema"];
  kaggle -> prep -> model; model -> clus; model -> eval;
  model -> st; model -> api; st -> db; prep -> pbi; clus -> pbi; prep -> clf; clf -> st; clf -> pbi;
}
""")

st.markdown("""
#### 🧰 Tech stack
| Layer | Technology |
|---|---|
| Data & ML | Python, pandas, NumPy, SciPy, scikit-learn (TF-IDF, cosine, TruncatedSVD, K-Means, t-SNE, One-vs-Rest Logistic Regression, MMR, Rocchio) |
| Frontend | Streamlit (multi-page), custom CSS, Plotly |
| Backend | FastAPI REST API, SQLite persistence |
| BI | Power BI Desktop (star schema, DAX measures, custom Netflix theme) |
| Deployment | GitHub + Streamlit Community Cloud |
| Quality | pytest unit and API tests |

#### 🚀 Future scope
- Collaborative filtering once real user ratings are collected (implicit-feedback ALS)
- Sentence-transformer embeddings for semantic search
- A/B testing of weight configurations using the like/dislike feedback table
""")
