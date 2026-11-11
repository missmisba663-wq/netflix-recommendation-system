import streamlit as st

from src.ui import get_titles, render_grid

df = get_titles()
st.title("🔍 Browse the Catalogue")

c1, c2, c3, c4 = st.columns([2.2, 1, 1.4, 1.2])
q = c1.text_input("Search title, cast or director", placeholder="e.g. Shah Rukh Khan, Narcos, Scorsese")
ctype = c2.selectbox("Type", ["All", "Movie", "TV Show"])
all_genres = sorted(df["genres"].str.split("|").explode().dropna().unique())
genre = c3.selectbox("Genre", ["All"] + all_genres)
sort = c4.selectbox("Sort by", ["Newest release", "Recently added", "Oldest release", "Title A–Z"])

c5, c6 = st.columns([1, 3])
countries = df["primary_country"].value_counts()
country = c5.selectbox("Country", ["All"] + [c for c in countries.index if c != "Unknown"])
ratings = c6.multiselect("Rating", sorted(df["rating"].unique()))

view = df
if q:
    ql = q.lower()
    view = view[view["title"].str.lower().str.contains(ql, regex=False)
                | view["cast"].str.lower().str.contains(ql, regex=False)
                | view["director"].str.lower().str.contains(ql, regex=False)]
if ctype != "All":
    view = view[view["type"] == ctype]
if genre != "All":
    view = view[view["genres"].str.contains(genre, regex=False)]
if country != "All":
    view = view[view["countries"].str.contains(country, regex=False)]
if ratings:
    view = view[view["rating"].isin(ratings)]
order = {"Newest release": ("release_year", False), "Recently added": ("date_added", False),
         "Oldest release": ("release_year", True), "Title A–Z": ("title", True)}[sort]
view = view.sort_values(order[0], ascending=order[1], na_position="last")

per_page = 24
pages = max(1, -(-len(view) // per_page))
top = st.columns([3, 1])
top[0].caption(f"**{len(view):,}** titles found")
page = top[1].number_input("Page", 1, pages, 1, label_visibility="collapsed")
render_grid(view.iloc[(page - 1) * per_page: page * per_page], key="browse", cols=6, show_score=False)
st.caption(f"Page {page} of {pages}")

with st.expander("📋 Table view / download"):
    cols = ["title", "type", "release_year", "rating", "duration", "listed_in", "country", "director"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=380)
    st.download_button("⬇️ Download CSV", view[cols].to_csv(index=False), "netflix_filtered.csv",
                       "text/csv")
