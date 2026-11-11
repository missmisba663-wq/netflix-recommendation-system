# 🎬 Netflix Movie & TV Show Recommendation System

An end-to-end **machine learning + full-stack + business intelligence** project built on the real
[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset from Kaggle
(8,800+ titles).

**Live app:** `https://<your-app-name>.streamlit.app` *(add your link after deploying)*

| Component | What it does |
|---|---|
| 🧠 **ML engine** | Hybrid content-based recommender (multi-field TF-IDF + weighted cosine), Rocchio taste profiles, MMR diversity re-ranking, natural-language search, K-Means clustering + t-SNE map, offline evaluation vs baselines |
| 🔮 **Supervised NLP** | AI Genre Predictor: multi-label genre tagger (One-vs-Rest Logistic Regression with per-genre thresholds tuned on a validation split), Movie-vs-TV and audience classifiers, word-level explanations, test-set F1 / accuracy / confusion matrices |
| 🎭 **Discovery** | Mood-based recommendations (10 moods) and an interactive 2-hop recommendation network graph |
| 🌐 **Website** | 8-page Streamlit web app with a Netflix-style UI, title cards, detail pop-ups, explanations ("why recommended"), live analytics, user profiles and a persistent watchlist |
| ⚙️ **Backend** | FastAPI REST API (`/recommend`, `/search`, `/titles`, `/stats`, watchlist CRUD) and SQLite database |
| 📊 **Power BI** | Star-schema data model, Netflix dark theme, 25+ DAX measures, 4-page report guide, embeddable in the website |
| ✅ **Testing** | pytest unit + API tests |

---

## 📁 Project structure
```
├── streamlit_app.py          # Website entry point (multi-page navigation)
├── app_pages/                # Home, Recommend, Genre Predictor, Browse, My List, Analytics, Model Insights, About
├── src/
│   ├── data_prep.py          # cleaning & feature engineering
│   ├── recommender.py        # ML recommendation engine
│   ├── clustering.py         # LSA + K-Means + t-SNE
│   ├── classifier.py         # supervised NLP: genre / type / audience models
│   ├── moods.py              # mood discovery + recommendation network graph
│   ├── evaluate.py           # offline evaluation vs baselines
│   ├── pipeline.py           # end-to-end pipeline + Power BI export
│   ├── db.py                 # SQLite watchlist / feedback
│   └── ui.py                 # shared UI components & theme
├── api/main.py               # FastAPI REST backend
├── tests/                    # pytest tests
├── data/raw/                 # Kaggle CSV
├── data/processed/           # cleaned data, clusters, evaluation results
├── powerbi/                  # Power BI data, theme, DAX, build guide
└── .streamlit/config.toml    # Netflix dark theme
```

## 🚀 Run locally
```bash
pip install -r requirements-dev.txt
python -m src.pipeline              # (optional) re-download, retrain, re-evaluate, re-export (~1.5 min)
streamlit run streamlit_app.py      # website  → http://localhost:8501
uvicorn api.main:app --reload       # REST API → http://127.0.0.1:8000/docs
pytest                              # tests
```

## 🧠 How the model works
Each title gets six TF-IDF feature blocks (description 1–2-grams, genres, cast, director, country,
meta = type + rating + decade). Blocks are L2-normalised and scaled by √wᶠ, so the dot product of two
title vectors is a **weighted sum of per-feature cosine similarities**:

  sim(a, b) = Σ wᶠ · cos(aᶠ, bᶠ),  weights = description .40, genres .25, cast .15, director .10, country .05, meta .05

- **Item-to-item:** top-k by similarity, with per-feature contribution and a text explanation.
- **Taste profile (Rocchio):** profile = mean(liked) − 0.5 · mean(disliked).
- **Search:** 0.6 · word TF-IDF + 0.4 · character n-gram TF-IDF, plus a bonus when the query names a nationality ("korean" → South Korea).
- **Diversity:** Maximal Marginal Relevance re-ranking.
- **Clustering:** TruncatedSVD (100-d LSA) → K-Means (k = 12) → t-SNE 2-D map.

### Supervised classifiers (held-out test set, 1,321 titles)
| Task | Model | Baseline |
|---|---|---|
| Genre tagging, micro-F1 (36 genres) | **46.9%** | 27.7% (always top-2 genres) |
| Movie vs TV Show, accuracy | **75.8%** | 69.6% (majority class) |
| Audience (4 classes), accuracy | **53.1%** | 44.2% (majority class) |

Input is only the title + a one-sentence synopsis; genres defined by origin (Korean, British) are the hardest.

### Recommender evaluation (400 random queries, top-10)
| Method | Genre Precision | People Hit | Plot Similarity | Diversity |
|---|---|---|---|---|
| Random | 24.6% | 0.1% | 0.4% | 99.6% |
| Genre-only | 100% | 2.4% | 1.0% | 98.8% |
| Description-only TF-IDF | 41.7% | 3.3% | 16.6% | 94.3% |
| **Hybrid (ours)** | **99.8%** | **12.8%** | 4.4% | 98.0% |
| **Hybrid + MMR (ours)** | **99.8%** | **12.8%** | 5.1% | 98.8% |

The hybrid model keeps near-perfect genre relevance while finding shared directors/actors **5×** more
often and with **4×** the plot similarity of the genre-only baseline.

## 🌍 Deploy to Streamlit Community Cloud
1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → *Sign in with GitHub* → **Create app**.
3. Repository = your repo, Branch = `main`, Main file = `streamlit_app.py`.
   *Advanced settings → Python 3.12.*
4. Click **Deploy**. In about 2 minutes you get a public `https://….streamlit.app` URL.
5. (Optional) *Settings → Secrets*: add `TMDB_API_KEY` (real posters) and `POWERBI_EMBED_URL`
   (see `.streamlit/secrets.toml.example`).

## 📊 Power BI
See [`powerbi/POWERBI_GUIDE.md`](powerbi/POWERBI_GUIDE.md) for the step-by-step build, and
[`powerbi/DAX_Measures.md`](powerbi/DAX_Measures.md) for all measures.

## 📜 Dataset licence
Netflix Movies and TV Shows by Shivam Bansal, Kaggle. See the dataset page for licence terms.
