"""Interactive dashboard for the Spotify & Billboard music trends analysis.
Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from dashboard_data import (CLUSTER_FEATURES, cluster_by_year, load_clean_songs,  # noqa: E402
                            popularity_correlations, yearly_trends)

st.set_page_config(page_title="HitScope", page_icon="🎵", layout="wide")
TYPE_COLORS = alt.Scale(domain=["High Energy", "Balanced", "Low Energy / Emotional"],
                        range=["#e45756", "#4c78a8", "#72b7b2"])
LABELS = {"danceability": "Danceability", "energy": "Energy", "valence": "Valence (positivity)",
          "acousticness": "Acousticness", "speechiness": "Speechiness", "loudness": "Loudness (dB)",
          "tempo": "Tempo (BPM)", "duration_min": "Song length (min)", "liveness": "Liveness",
          "instrumentalness": "Instrumentalness", "popularity": "Popularity"}


@st.cache_data
def load():
    songs = load_clean_songs(ROOT / "data" / "songs_normalize.csv")
    clustered, loadings = cluster_by_year(songs)
    matches = pd.read_csv(ROOT / "data" / "billboard_matches.csv", parse_dates=["first_chart_date"])
    return songs, clustered, loadings, yearly_trends(songs), popularity_correlations(songs), matches


songs, clustered, loadings, trends, corr, matches = load()

st.title("🎵 HitScope")
st.subheader("Two decades of hit music, analyzed")
st.caption(f"How {len(songs):,} Spotify top hits (1999–2019) changed over time, what makes them popular, "
           "and how Spotify success compares with the Billboard Hot 100.")

first, last = trends.iloc[0], trends.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Songs analyzed", f"{len(songs):,}")
c1.caption(f"by {songs['artist'].nunique():,} artists")
c2.metric("Avg song length in 2019", f"{last.duration_min:.1f} min")
c2.caption(f"{(first.duration_min - last.duration_min) * 60:.0f} seconds shorter than in {int(first.year)}")
c3.metric("Charted on Billboard Hot 100", f"{matches['charted'].mean():.0%}")
c3.caption(f"{matches['charted'].sum():,} of {len(matches):,} songs")
c4.metric("Strongest link to popularity", f"r = {corr['correlation'].abs().max():.2f}")
c4.caption("no single audio feature explains popularity")

tab_trends, tab_pop, tab_types, tab_bb, tab_song = st.tabs(
    ["📈 Trends over time", "🎯 What drives popularity?", "🎧 Song types", "📊 Spotify vs Billboard", "🔍 Song explorer"])

# ------------------------------------------------------------------ trends
with tab_trends:
    st.subheader("How the sound of hit music changed")
    picks = st.multiselect("Features to compare", [f for f in LABELS if f in trends.columns],
                           default=["danceability", "energy", "acousticness"],
                           format_func=LABELS.get)
    smooth = st.toggle("Smooth with a 3-year rolling average", value=True,
                       help="Averages each year with its neighbors so the long-term direction is easier to see.")
    if picks:
        scaled = trends[["year"] + picks].copy()
        if smooth:
            scaled[picks] = scaled[picks].rolling(3, center=True, min_periods=1).mean()
        for f in picks:  # scale each feature 0-1 so different units fit on one chart
            lo, hi = scaled[f].min(), scaled[f].max()
            scaled[f] = (scaled[f] - lo) / (hi - lo) if hi > lo else 0.5
        long = scaled.melt("year", var_name="feature", value_name="scaled")
        long["feature"] = long["feature"].map(LABELS)
        st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("scaled:Q", title="Relative level (0 = lowest year, 1 = highest year)"),
            color=alt.Color("feature:N", title=None, scale=alt.Scale(scheme="tableau10")),
            strokeWidth=alt.value(3),
            tooltip=["year", "feature", alt.Tooltip("scaled:Q", format=".2f")],
        ).properties(height=380), width="stretch")
        st.caption("Each feature is scaled to its own 0–1 range so they can be compared on one chart.")
    one = st.selectbox("See actual values for", list(LABELS), index=7, format_func=LABELS.get)
    st.altair_chart(alt.Chart(trends).mark_area(line=True, opacity=0.3).encode(
        x=alt.X("year:O", title="Year"), y=alt.Y(f"{one}:Q", title=LABELS[one], scale=alt.Scale(zero=False)),
        tooltip=["year", alt.Tooltip(f"{one}:Q", format=".3f")]).properties(height=250), width="stretch")

# ------------------------------------------------------------------ popularity
with tab_pop:
    st.subheader("Can audio features predict a hit's popularity?")
    st.write("Short answer: not on their own. Every feature's correlation with popularity is close to zero, "
             "so popularity depends on a combination of factors (and on things audio can't measure).")
    corr_plot = corr.assign(feature=corr["feature"].map(LABELS))
    st.altair_chart(alt.Chart(corr_plot).mark_bar().encode(
        x=alt.X("correlation:Q", title="Correlation with popularity", scale=alt.Scale(domain=[-0.2, 0.2])),
        y=alt.Y("feature:N", sort="-x", title=None),
        color=alt.condition("datum.correlation > 0", alt.value("#4c78a8"), alt.value("#e45756")),
        tooltip=["feature", alt.Tooltip("correlation:Q", format=".3f")]).properties(height=320), width="stretch")
    f = st.selectbox("Plot a feature against popularity", CLUSTER_FEATURES + ["acousticness", "duration_min"],
                     format_func=LABELS.get)
    st.altair_chart(alt.Chart(songs).mark_circle(size=35, opacity=0.4).encode(
        x=alt.X(f"{f}:Q", title=LABELS[f], scale=alt.Scale(zero=False)),
        y=alt.Y("popularity:Q", title="Popularity"),
        tooltip=["song", "artist", "year", f, "popularity"]).properties(height=360), width="stretch")

# ------------------------------------------------------------------ song types
with tab_types:
    st.subheader("Three types of hit songs, found with K-Means and PCA")
    st.write("For each year, songs are grouped into 3 clusters using 6 audio features "
             "(danceability, energy, loudness, tempo, liveness, valence). PCA then compresses those "
             "6 features into 2 dimensions so the clusters can be plotted.")
    years = sorted(clustered["year"].unique())
    year = st.select_slider("Year", options=years, value=2010)
    yd = clustered[clustered["year"] == year]
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(alt.Chart(yd).mark_circle(size=70, opacity=0.8).encode(
            x=alt.X("PC1:Q", title="PC1: Intensity (energy, loudness) →"),
            y=alt.Y("PC2:Q", title="PC2: Emotion / rhythm (valence, danceability) →"),
            color=alt.Color("song_type:N", title="Song type", scale=TYPE_COLORS),
            tooltip=["song", "artist", "song_type", "popularity", "energy", "valence", "danceability"],
        ).properties(height=420, title=f"Hit songs of {year}"), width="stretch")
    with right:
        load_tbl = loadings[year].rename(index=LABELS).round(2)
        ev = loadings[year].attrs["explained"]
        st.markdown(f"**What the two axes mean in {year}**")
        st.dataframe(load_tbl, width="stretch")
        st.caption(f"PC1 explains {ev[0]:.0%} and PC2 {ev[1]:.0%} of the variation in the 6 features. "
                   "Larger numbers mean that feature drives the axis more.")
        prof = yd.groupby("song_type")[["energy", "loudness", "valence", "danceability", "popularity"]].mean()
        prof.insert(0, "songs", yd["song_type"].value_counts())
        st.markdown("**Average profile of each type**")
        st.dataframe(prof.round(2), width="stretch")
    share = clustered.groupby(["year", "song_type"]).size().reset_index(name="songs")
    share["share"] = share["songs"] / share.groupby("year")["songs"].transform("sum")
    st.altair_chart(alt.Chart(share).mark_bar().encode(
        x=alt.X("year:O", title="Year"), y=alt.Y("share:Q", title="Share of that year's hits", axis=alt.Axis(format="%")),
        color=alt.Color("song_type:N", title="Song type", scale=TYPE_COLORS),
        tooltip=["year", "song_type", "songs", alt.Tooltip("share:Q", format=".0%")],
    ).properties(height=280, title="Mix of song types each year"), width="stretch")
    pop_by_type = clustered.groupby("song_type")["popularity"].mean()
    st.caption("Average popularity by type: " + ", ".join(f"{k} {v:.1f}" for k, v in pop_by_type.items())
               + ". No type is clearly more popular, so hits come from all three.")

# ------------------------------------------------------------------ billboard
with tab_bb:
    st.subheader("Do Spotify hits also chart on the Billboard Hot 100?")
    ch = matches[matches["charted"]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Charted on the Hot 100", f"{matches['charted'].mean():.0%}")
    m1.caption(f"{len(ch):,} of {len(matches):,} songs")
    m2.metric("Reached the top 10 (of those that charted)", f"{(ch['peak_rank'] <= 10).mean():.0%}")
    m3.metric("Median weeks on the chart", f"{ch['weeks_on_chart'].median():.0f}")
    l, r = st.columns(2)
    l.altair_chart(alt.Chart(ch).mark_bar().encode(
        x=alt.X("peak_rank:Q", bin=alt.Bin(step=10), title="Best Billboard rank (1 = #1 hit)"),
        y=alt.Y("count():Q", title="Songs"), tooltip=[alt.Tooltip("count():Q", title="Songs")],
    ).properties(height=280, title="How high did they get?"), width="stretch")
    by_year = matches.groupby("year")["charted"].agg(["mean", "size"]).reset_index()
    by_year = by_year[by_year["size"] >= 20]
    r.altair_chart(alt.Chart(by_year).mark_line(point=True).encode(
        x=alt.X("year:O", title="Year"), y=alt.Y("mean:Q", title="Share that charted", axis=alt.Axis(format="%"),
                                                 scale=alt.Scale(domain=[0, 1])),
        tooltip=["year", alt.Tooltip("mean:Q", format=".0%", title="charted"), alt.Tooltip("size:Q", title="songs")],
    ).properties(height=280, title="Share of each year's Spotify hits that charted"), width="stretch")
    pop_cmp = matches.groupby("charted")["popularity"].mean()
    st.caption(f"Average Spotify popularity: {pop_cmp.get(True, 0):.1f} for songs that charted vs "
               f"{pop_cmp.get(False, 0):.1f} for songs that didn't. Many songs that never charted are UK and "
               "European hits, which makes sense since Billboard measures the US market.")
    st.markdown("**Look up any song**")
    q = st.text_input("Search by song or artist", placeholder="e.g. Rihanna")
    view = matches.copy()
    if q:
        view = view[view["song"].str.contains(q, case=False) | view["artist"].str.contains(q, case=False)]
    st.dataframe(view.sort_values("popularity", ascending=False).head(300), hide_index=True, width="stretch",
                 column_config={"charted": "On Hot 100?", "peak_rank": "Best rank", "weeks_on_chart": "Weeks",
                                "first_chart_date": st.column_config.DateColumn("First charted"),
                                "popularity": st.column_config.ProgressColumn("Spotify popularity", min_value=0, max_value=100, format="%d")})

# ------------------------------------------------------------------ song explorer
with tab_song:
    st.subheader("How does one song compare to its year?")
    pick = st.selectbox("Song", clustered.sort_values("popularity", ascending=False).apply(
        lambda r: f"{r.song} — {r.artist} ({r.year})", axis=1).tolist())
    row = clustered[clustered.apply(lambda r: f"{r.song} — {r.artist} ({r.year})", axis=1) == pick].iloc[0]
    yr = clustered[clustered["year"] == row.year]
    bb = matches[(matches["song"] == row.song) & (matches["artist"] == row.artist)].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Song type", row.song_type)
    b.metric("Spotify popularity", int(row.popularity))
    b.caption(f"{row.popularity - yr['popularity'].mean():+.0f} vs the average {row.year} hit")
    c.metric("Billboard Hot 100", f"#{int(bb.peak_rank)} peak" if bb.charted else "Did not chart")
    if bb.charted:
        c.caption(f"{int(bb.weeks_on_chart)} weeks on the chart")
    feats = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness"]
    cmp = pd.DataFrame({"feature": [LABELS[f] for f in feats] * 2,
                        "value": [row[f] for f in feats] + [yr[f].mean() for f in feats],
                        "who": ["This song"] * len(feats) + [f"Average {row.year} hit"] * len(feats)})
    st.altair_chart(alt.Chart(cmp).mark_bar().encode(
        y=alt.Y("feature:N", title=None), x=alt.X("value:Q", title="Value (0–1)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("who:N", title=None), yOffset="who:N",
        tooltip=["who", "feature", alt.Tooltip("value:Q", format=".2f")]).properties(height=320), width="stretch")

st.divider()
st.caption("Data: Top Hits Spotify 2000–2019 and Billboard Hot 100 (Kaggle). "
           "Built with Python, pandas, scikit-learn, Altair, and Streamlit.")
