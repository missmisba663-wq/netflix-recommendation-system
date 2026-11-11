import calendar

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import RED, TYPE_COLORS, get_secret, get_titles, kpi

df = get_titles()
st.title("📊 Netflix Content Analytics")
st.caption("Exploratory data analysis of the Kaggle Netflix catalogue. The same data model powers "
           "the Power BI dashboard (see the bottom of this page).")

# --------------------------------------------------------------- filters (one row)
f1, f2, f3 = st.columns([1, 2, 1.4])
ctype = f1.selectbox("Type", ["All", "Movie", "TV Show"])
ya = df["year_added"].dropna().astype(int)
yr = f2.slider("Year added to Netflix", int(ya.min()), int(ya.max()), (2015, int(ya.max())))
aud = f3.multiselect("Audience", ["Kids", "Older Kids", "Teens", "Adults", "Unrated"])

v = df[df["year_added"].between(*yr)]
if ctype != "All":
    v = v[v["type"] == ctype]
if aud:
    v = v[v["maturity_group"].isin(aud)]
if v.empty:
    st.warning("No titles match these filters.")
    st.stop()

# --------------------------------------------------------------- KPIs
c = st.columns(6)
kpi(c[0], f"{len(v):,}", "Titles")
kpi(c[1], f"{(v['type'] == 'Movie').mean():.0%}", "Movies share")
kpi(c[2], f"{v['duration_min'].mean():.0f} min" if v["duration_min"].notna().any() else "–", "Avg movie length")
kpi(c[3], f"{v['seasons'].mean():.1f}" if v["seasons"].notna().any() else "–", "Avg seasons")
kpi(c[4], f"{v['countries'].str.split('|').explode().replace('', None).nunique()}", "Countries")
lag = int(v["years_to_netflix"].median())
kpi(c[5], f"{lag} yr{'s' if lag != 1 else ''}", "Median lag to Netflix")
st.write("")

def explode(col):
    return v.assign(_x=v[col].str.split("|")).explode("_x").query("_x != ''")

# --------------------------------------------------------------- growth
a, b = st.columns([1.6, 1])
with a:
    g = v.dropna(subset=["year_added"]).groupby(["year_added", "type"]).size().reset_index(name="Titles")
    fig = px.line(g, x="year_added", y="Titles", color="type", markers=True,
                  color_discrete_map=TYPE_COLORS, title="Titles added per year")
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
with b:
    m = (v.dropna(subset=["year_added"]).groupby(["year_added", "month_added"]).size()
         .reset_index(name="Titles"))
    m["Month"] = m["month_added"].astype(int).map(lambda i: calendar.month_abbr[i])
    fig = px.density_heatmap(m, x="Month", y="year_added", z="Titles", histfunc="sum",
                             color_continuous_scale=["#2a0a0c", "#7a0a10", "#c40812", "#ff4d55", "#ffc2c5"],
                             category_orders={"Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                                        "Aug", "Sep", "Oct", "Nov", "Dec"]},
                             title="When titles are added (month × year)")
    fig.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_colorbar_title=None)
    fig.update_yaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- genres & countries
a, b = st.columns(2)
with a:
    gg = explode("genres")["_x"].value_counts().head(12).sort_values()
    fig = px.bar(x=gg.values, y=gg.index, orientation="h", title="Top 12 genres", text=gg.values)
    fig.update_traces(marker_color=RED, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_showgrid=False, xaxis_showticklabels=False,
                      xaxis_range=[0, gg.max() * 1.15])
    st.plotly_chart(fig, width="stretch")
with b:
    cc = explode("countries")["_x"].value_counts().reset_index()
    cc.columns = ["country", "Titles"]
    fig = px.choropleth(cc, locations="country", locationmode="country names", color="Titles",
                        color_continuous_scale=["#3a0a0d", "#8f0a12", "#E50914", "#ff6b72", "#ffd1d3"],
                        title="Where the content comes from", hover_name="country")
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False,
                    landcolor="#262626", projection_type="natural earth", showcountries=False)
    fig.update_layout(coloraxis_colorbar_title=None, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- ratings & durations
a, b, c3 = st.columns(3)
with a:
    r = v.groupby(["rating", "type"]).size().reset_index(name="Titles")
    order = v["rating"].value_counts().index.tolist()
    fig = px.bar(r, x="rating", y="Titles", color="type", barmode="group",
                 color_discrete_map=TYPE_COLORS, category_orders={"rating": order},
                 title="Maturity ratings")
    fig.update_traces(marker_line_width=2, marker_line_color="#141414")
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")
with b:
    mv = v[v["type"] == "Movie"]
    fig = px.histogram(mv, x="duration_min", nbins=40, title="Movie runtime (minutes)")
    fig.update_traces(marker_color=RED, marker_line_width=2, marker_line_color="#141414")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
with c3:
    tv = v[v["type"] == "TV Show"]["seasons"].value_counts().sort_index().head(10)
    fig = px.bar(x=tv.index.astype(int).astype(str), y=tv.values, title="TV shows by number of seasons")
    fig.update_traces(marker_color="#3987e5")
    fig.update_layout(xaxis_title="Seasons", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- people & age
a, b, c3 = st.columns(3)
with a:
    d = explode("directors")["_x"].value_counts().head(10).sort_values()
    fig = px.bar(x=d.values, y=d.index, orientation="h", title="Most frequent directors", text=d.values)
    fig.update_traces(marker_color=RED, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_showticklabels=False, xaxis_showgrid=False,
                      xaxis_range=[0, d.max() * 1.18])
    st.plotly_chart(fig, width="stretch")
with b:
    ac = explode("cast_list")["_x"].value_counts().head(10).sort_values()
    fig = px.bar(x=ac.values, y=ac.index, orientation="h", title="Most frequent actors", text=ac.values)
    fig.update_traces(marker_color=RED, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_showticklabels=False, xaxis_showgrid=False,
                      xaxis_range=[0, ac.max() * 1.18])
    st.plotly_chart(fig, width="stretch")
with c3:
    mg = v["maturity_group"].value_counts().reindex(["Kids", "Older Kids", "Teens", "Adults", "Unrated"]).dropna()
    fig = px.bar(x=mg.index, y=mg.values, title="Target audience", text=mg.values)
    fig.update_traces(marker_color=RED, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_showticklabels=False, yaxis_showgrid=False,
                      yaxis_range=[0, mg.max() * 1.15])
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- decades
dec = v.groupby(["decade", "type"]).size().reset_index(name="Titles")
fig = px.bar(dec, x="decade", y="Titles", color="type", color_discrete_map=TYPE_COLORS,
             title="Catalogue by release decade")
fig.update_traces(marker_line_width=2, marker_line_color="#141414")
fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
st.plotly_chart(fig, width="stretch")

with st.expander("📋 Data table"):
    st.dataframe(v[["title", "type", "release_year", "year_added", "rating", "primary_genre",
                    "primary_country"]], hide_index=True, width="stretch", height=320)

# --------------------------------------------------------------- Power BI
st.divider()
st.subheader("📈 Power BI Dashboard")
pbi = get_secret("POWERBI_EMBED_URL")
if pbi:
    st.components.v1.iframe(pbi, height=640, scrolling=False)
else:
    st.info("The interactive Power BI report will appear here once it is published. Build it from "
            "`powerbi/` (see `powerbi/POWERBI_GUIDE.md`), use **File → Embed report → Publish to web**, "
            "and add the link as `POWERBI_EMBED_URL` in the Streamlit secrets.")
