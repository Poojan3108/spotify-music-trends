# Data preparation for the interactive dashboard (app.py).
# Reuses the same analysis choices as the original pipeline:
#   - the 6 audio features used for clustering
#   - year-by-year StandardScaler + K-Means (k=3, random_state=42) + PCA (2 components)
#   - text cleaning to match Spotify songs with Billboard Hot 100 entries

import re

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ['danceability', 'energy', 'loudness', 'tempo', 'liveness', 'valence']
TREND_FEATURES = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness',
                  'loudness', 'tempo', 'duration_min']
MIN_SONGS_PER_YEAR = 20   # same rule as the original pipeline


# ---------------------------------------------------------------- loading
def load_clean_songs(path):
    """Load the Spotify top-hits data and actually apply the cleaning steps."""
    songs = pd.read_csv(path)
    songs = songs.drop_duplicates()                                   # exact duplicates
    songs = songs.drop_duplicates(subset=['song', 'artist'], keep='first')  # same song re-listed
    songs['year'] = songs['year'].astype(int)
    songs['duration_min'] = songs['duration_ms'] / 60000
    songs['main_genre'] = songs['genre'].str.split(',').str[0].str.strip()
    return songs.reset_index(drop=True)


# ---------------------------------------------------------------- trends
def yearly_trends(songs):
    years = songs[songs.groupby('year')['year'].transform('size') >= MIN_SONGS_PER_YEAR]
    return years.groupby('year')[TREND_FEATURES + ['popularity']].mean().reset_index()


def popularity_correlations(songs):
    numeric = songs[CLUSTER_FEATURES + ['acousticness', 'speechiness', 'duration_min',
                                        'instrumentalness', 'popularity']]
    corr = numeric.corr()['popularity'].drop('popularity')
    return corr.sort_values().reset_index().rename(columns={'index': 'feature', 'popularity': 'correlation'})


# ---------------------------------------------------------------- clusters + PCA
def name_clusters(data):
    """Label the 3 clusters by intensity: highest energy+loudness = High Energy, lowest = Low Energy / Emotional."""
    z = data[['energy', 'loudness']].apply(lambda c: (c - c.mean()) / c.std())
    intensity = z.mean(axis=1).groupby(data['cluster']).mean().sort_values()
    names = dict(zip(intensity.index, ['Low Energy / Emotional', 'Balanced', 'High Energy']))
    return data['cluster'].map(names)


def cluster_by_year(songs):
    """K-Means (k=3) and PCA (2 components) fitted separately for each year, as in the original analysis."""
    frames, loadings = [], {}
    for year in sorted(songs['year'].unique()):
        data = songs[songs['year'] == year].copy()
        if len(data) < MIN_SONGS_PER_YEAR:
            continue
        X = StandardScaler().fit_transform(data[CLUSTER_FEATURES])
        data['cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
        pca = PCA(n_components=2).fit(X)
        pcs = pca.transform(X)
        # orient components so PC1 increases with energy and PC2 with valence (signs are arbitrary in PCA)
        e, v = CLUSTER_FEATURES.index('energy'), CLUSTER_FEATURES.index('valence')
        sign1 = 1 if pca.components_[0][e] >= 0 else -1
        sign2 = 1 if pca.components_[1][v] >= 0 else -1
        data['PC1'], data['PC2'] = pcs[:, 0] * sign1, pcs[:, 1] * sign2
        data['song_type'] = name_clusters(data)
        loadings[year] = pd.DataFrame(
            {'PC1': pca.components_[0] * sign1, 'PC2': pca.components_[1] * sign2},
            index=CLUSTER_FEATURES)
        loadings[year].attrs['explained'] = pca.explained_variance_ratio_.tolist()
        frames.append(data)
    return pd.concat(frames, ignore_index=True), loadings


# ---------------------------------------------------------------- Billboard matching
def clean_song_title(text):
    """'Love The Way You Lie (feat. Rihanna) - Radio Edit' -> 'love the way you lie'"""
    text = str(text).lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    text = text.split(" - ")[0].replace("&", "and")
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def clean_artist(text):
    """'Drake Featuring Rihanna' / 'Jay-Z & Kanye West' -> lead artist only, normalized"""
    text = str(text).lower()
    text = re.split(r"\s+(?:featuring|feat\.?|ft\.?|with|x|&|,|and)\s+", text)[0]
    return re.sub(r"[^a-z0-9 ]", "", text.replace("&", "and")).strip()


def match_billboard(songs, charts, since="1998-01-01"):
    """For every Spotify song: did it chart on the Billboard Hot 100, and how well?"""
    charts = charts[charts['date'] >= since].copy()
    charts['song_key'] = charts['song'].map(clean_song_title)
    charts['artist_key'] = charts['artist'].map(clean_artist)
    charts['artist_full'] = charts['artist'].map(lambda t: re.sub(r"[^a-z0-9 ]", "", str(t).lower()))

    songs = songs[['song', 'artist', 'year', 'popularity']].copy()
    songs['song_key'] = songs['song'].map(clean_song_title)
    songs['artist_key'] = songs['artist'].map(clean_artist)

    by_song = {k: g for k, g in charts.groupby('song_key')}
    rows = []
    for s in songs.itertuples(index=False):
        g = by_song.get(s.song_key)
        if g is not None:
            g = g[(g['artist_key'] == s.artist_key) |
                  g['artist_full'].str.contains(re.escape(s.artist_key), regex=True)]
        if g is None or g.empty:
            rows.append({'charted': False})
            continue
        rows.append({'charted': True, 'peak_rank': int(g['rank'].min()),
                     'weeks_on_chart': int(g['date'].nunique()),
                     'first_chart_date': g['date'].min()})
    result = pd.concat([songs[['song', 'artist', 'year', 'popularity']].reset_index(drop=True),
                        pd.DataFrame(rows)], axis=1)
    return result
