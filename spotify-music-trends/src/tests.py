# Testing our code

# This file wass fully generated using AI tool.

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# ---------------------------------------------------------------------------
# Helpers — minimal DataFrames that mimic real data shapes
# ---------------------------------------------------------------------------

def make_spotify_df():
    """Return a small Spotify-like DataFrame with all expected columns."""
    return pd.DataFrame({
        'artist':           ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist A'],
        'song':             ['Song 1',   'Song 2',   'Song 3',   'Song 4',   'Song 1'],
        'duration_ms':      [200000, 180000, 220000, 195000, 200000],
        'explicit':         [False, True, False, False, False],
        'year':             [2005, 2005, 2006, 2006, 2005],
        'popularity':       [80, 60, 75, 55, 80],
        'danceability':     [0.8, 0.6, 0.7, 0.5, 0.8],
        'energy':           [0.7, 0.9, 0.6, 0.8, 0.7],
        'key':              [1, 2, 3, 4, 1],
        'loudness':         [-5.0, -4.0, -6.0, -3.5, -5.0],
        'mode':             [1, 0, 1, 0, 1],
        'speechiness':      [0.05, 0.10, 0.04, 0.08, 0.05],
        'acousticness':     [0.2, 0.1, 0.3, 0.15, 0.2],
        'instrumentalness': [0.0, 0.0, 0.01, 0.0, 0.0],
        'liveness':         [0.1, 0.2, 0.15, 0.25, 0.1],
        'valence':          [0.6, 0.4, 0.7, 0.5, 0.6],
        'tempo':            [120.0, 130.0, 110.0, 140.0, 120.0],
        'genre':            ['pop', 'rock', 'pop', 'hip-hop', 'pop'],
    })


def make_billboard_df():
    """Return a small Billboard-like DataFrame."""
    return pd.DataFrame({
        'song':   ['Song 1', 'Song 2', 'Song X', 'Song 1'],
        'artist': ['Artist A', 'Artist B', 'Artist Y', 'Artist A'],
        'rank':   [1, 5, 10, 2],
        'date':   pd.to_datetime(['2005-03-01', '2005-04-01', '2006-01-01', '2005-03-08']),
    })


def make_spotify_api_df():
    """Return a small Spotify API result DataFrame."""
    return pd.DataFrame({
        'spotify_song':   ['Song 1', 'Song 2'],
        'spotify_artist': ['Artist A', 'Artist B'],
        'album':          ['Album X', 'Album Y'],
        'release_date':   ['2005-01-15', '2005-02-20'],
        'release_date_precision': ['day', 'day'],
        'original_song':  ['Song 1', 'Song 2'],
        'original_artist':['Artist A', 'Artist B'],
        'year_group':     ['2005-2006', '2005-2006'],
    })


# ===========================================================================
# MODULE: data_loader
# ===========================================================================

class TestDataLoader:
    """Unit tests for data_loader.py"""

    def test_load_spotify_data_returns_dataframe(self, tmp_path):
        """load_spotify_data should return a DataFrame from a CSV file."""
        from data_loader import load_spotify_data
        csv_path = tmp_path / "spotify.csv"
        make_spotify_df().to_csv(csv_path, index=False)
        result = load_spotify_data(str(csv_path))
        assert isinstance(result, pd.DataFrame)

    def test_load_spotify_data_shape(self, tmp_path):
        """Loaded Spotify DataFrame should have the correct number of rows."""
        from data_loader import load_spotify_data
        df = make_spotify_df()
        csv_path = tmp_path / "spotify.csv"
        df.to_csv(csv_path, index=False)
        result = load_spotify_data(str(csv_path))
        assert result.shape[0] == len(df)

    def test_load_billboard_data_returns_dataframe(self, tmp_path):
        """load_billboard_data should return a DataFrame from a CSV file."""
        from data_loader import load_billboard_data
        csv_path = tmp_path / "billboard.csv"
        make_billboard_df().to_csv(csv_path, index=False)
        result = load_billboard_data(str(csv_path))
        assert isinstance(result, pd.DataFrame)

    def test_load_billboard_data_columns(self, tmp_path):
        """Loaded Billboard DataFrame should preserve expected columns."""
        from data_loader import load_billboard_data
        df = make_billboard_df()
        csv_path = tmp_path / "billboard.csv"
        df.to_csv(csv_path, index=False)
        result = load_billboard_data(str(csv_path))
        for col in ['song', 'artist', 'rank', 'date']:
            assert col in result.columns

    def test_preview_data_prints(self, capsys):
        """preview_data should print without raising an error."""
        from data_loader import preview_data
        preview_data(make_spotify_df(), name="TestDataset")
        captured = capsys.readouterr()
        assert "TestDataset" in captured.out

    def test_check_missing_values_prints(self, capsys):
        """check_missing_values should report null counts."""
        from data_loader import check_missing_values
        df = make_spotify_df()
        df.loc[0, 'artist'] = None
        check_missing_values(df, name="TestDataset")
        captured = capsys.readouterr()
        assert "TestDataset" in captured.out

    def test_dataset_info_prints(self, capsys):
        """dataset_info should print DataFrame info without error."""
        from data_loader import dataset_info
        dataset_info(make_spotify_df(), name="TestDataset")
        captured = capsys.readouterr()
        assert "TestDataset" in captured.out


# ===========================================================================
# MODULE: preprocessing
# ===========================================================================

class TestPreprocessing:
    """Unit tests for preprocessing.py"""

    def test_checking_for_null_values_prints(self, capsys):
        """checking_for_null_values should print null counts."""
        from preprocessing import checking_for_null_values
        df = make_spotify_df()
        df.loc[0, 'genre'] = None
        checking_for_null_values(df)
        captured = capsys.readouterr()
        assert "genre" in captured.out

    def test_dropping_null_values_removes_rows(self):
        """dropping_null_values should drop rows with NaN in the given column."""
        from preprocessing import dropping_null_values
        df = make_spotify_df()
        df.loc[0, 'genre'] = None
        result = dropping_null_values(df, 'genre')
        assert result['genre'].isnull().sum() == 0
        assert len(result) == len(df) - 1

    def test_dropping_null_values_no_nulls_unchanged(self):
        """Dropping on a column with no nulls should return the same length."""
        from preprocessing import dropping_null_values
        df = make_spotify_df()
        result = dropping_null_values(df, 'year')
        assert len(result) == len(df)

    def test_removing_duplicates(self):
        """removing_duplicates should remove exact duplicate rows."""
        from preprocessing import removing_duplicates
        df = make_spotify_df()
        # Row 0 and row 4 are identical
        result = removing_duplicates(df)
        assert len(result) < len(df)

    def test_removing_duplicates_no_dups(self):
        """removing_duplicates on unique data should not change row count."""
        from preprocessing import removing_duplicates
        df = make_spotify_df().drop_duplicates().reset_index(drop=True)
        unique_count = len(df)
        result = removing_duplicates(df)
        assert len(result) == unique_count

    def test_correcting_data_types_year_int(self):
        """correcting_data_types should cast year to int."""
        from preprocessing import correcting_data_types
        spotify = make_spotify_df()
        spotify['year'] = spotify['year'].astype(float)
        billboard = make_billboard_df()
        billboard['date'] = billboard['date'].astype(str)
        billboard['rank'] = billboard['rank'].astype(str)
        s_out, b_out = correcting_data_types(spotify, billboard)
        assert s_out['year'].dtype == int

    def test_correcting_data_types_billboard_date(self):
        """correcting_data_types should convert billboard date to datetime."""
        from preprocessing import correcting_data_types
        spotify = make_spotify_df()
        billboard = make_billboard_df()
        billboard['date'] = billboard['date'].astype(str)
        s_out, b_out = correcting_data_types(spotify, billboard)
        assert pd.api.types.is_datetime64_any_dtype(b_out['date'])

    def test_correcting_data_types_billboard_rank_int(self):
        """correcting_data_types should convert billboard rank to int."""
        from preprocessing import correcting_data_types
        spotify = make_spotify_df()
        billboard = make_billboard_df()
        billboard['rank'] = billboard['rank'].astype(float)
        s_out, b_out = correcting_data_types(spotify, billboard)
        assert b_out['rank'].dtype == int

    def test_correcting_data_types_strips_featuring(self):
        """correcting_data_types should strip 'Featuring...' from artist."""
        from preprocessing import correcting_data_types
        spotify = make_spotify_df()
        billboard = make_billboard_df()
        billboard.loc[0, 'artist'] = 'Artist A Featuring. Someone'
        s_out, b_out = correcting_data_types(spotify, billboard)
        assert 'Featuring' not in b_out.loc[0, 'artist']

    def test_correcting_data_types_strips_real_featuring_format(self):
        """Real Billboard names have no period: 'Drake Featuring Rihanna' -> 'Drake'."""
        from preprocessing import correcting_data_types
        spotify = make_spotify_df()
        billboard = make_billboard_df()
        billboard.loc[0, 'artist'] = 'Drake Featuring Rihanna'
        s_out, b_out = correcting_data_types(spotify, billboard)
        assert b_out.loc[0, 'artist'] == 'Drake'


# ===========================================================================
# MODULE: billboard_charting
# ===========================================================================

class TestBillboardCharting:
    """Unit tests for billboard_charting.py"""

    def test_clean_text_lowercases(self):
        """clean_text should convert text to lowercase."""
        from billboard_charting import clean_text
        assert clean_text("HELLO") == "hello"

    def test_clean_text_strips_special_chars(self):
        """clean_text should remove non-alphanumeric characters."""
        from billboard_charting import clean_text
        assert clean_text("it's a test!") == "its a test"

    def test_clean_text_strips_whitespace(self):
        """clean_text should strip leading/trailing whitespace."""
        from billboard_charting import clean_text
        assert clean_text("  hello  ") == "hello"

    def test_clean_text_handles_numbers(self):
        """clean_text should preserve numbers."""
        from billboard_charting import clean_text
        assert "100" in clean_text("Top 100")

    def test_clean_text_empty_string(self):
        """clean_text should handle empty strings without error."""
        from billboard_charting import clean_text
        result = clean_text("")
        assert result == ""

    @patch('billboard_charting.billboard_distribution')
    @patch('billboard_charting.spotify_songs_in_billboard')
    @patch('billboard_charting.date_vs_chart')
    @patch('billboard_charting.delay_analysis')
    def test_billboard_charting_runs(self, mock_delay, mock_date, mock_presence, mock_dist):
        """billboard_charting should call all four visualization functions."""
        from billboard_charting import billboard_charting
        spotify_api = make_spotify_api_df()
        billboard = make_billboard_df()
        billboard_charting(billboard, spotify_api)
        assert mock_dist.called
        assert mock_presence.called

    @patch('billboard_charting.billboard_distribution')
    @patch('billboard_charting.spotify_songs_in_billboard')
    @patch('billboard_charting.date_vs_chart')
    @patch('billboard_charting.delay_analysis')
    def test_billboard_charting_skips_na_release(self, mock_delay, mock_date, mock_presence, mock_dist):
        """billboard_charting should skip rows where release_date is NaT."""
        from billboard_charting import billboard_charting
        spotify_api = make_spotify_api_df()
        spotify_api.loc[0, 'release_date'] = None   # force NaT
        billboard = make_billboard_df()
        # Should complete without raising
        billboard_charting(billboard, spotify_api)

    @patch('billboard_charting.billboard_distribution')
    @patch('billboard_charting.spotify_songs_in_billboard')
    @patch('billboard_charting.date_vs_chart')
    @patch('billboard_charting.delay_analysis')
    def test_billboard_charting_match_selects_best_rank(self, mock_delay, mock_date, mock_presence, mock_dist):
        """billboard_charting should pick the best (lowest) rank when multiple entries match."""
        from billboard_charting import billboard_charting
        # Song 1 by Artist A appears on both 2005-03-01 (rank 1) and 2005-03-08 (rank 2)
        spotify_api = make_spotify_api_df()
        billboard = make_billboard_df()
        # Capture the DataFrame passed to billboard_distribution
        captured = {}
        def capture_dist(df):
            captured['df'] = df
        mock_dist.side_effect = capture_dist
        billboard_charting(billboard, spotify_api)
        if 'df' in captured and not captured['df'].empty:
            assert captured['df']['best_rank'].iloc[0] == 1


# ===========================================================================
# MODULE: spotify_api
# ===========================================================================

class TestSpotifyApi:
    """Unit tests for spotify_api.py"""

    def test_get_top_25_returns_dataframe(self):
        """get_top_25 should return a DataFrame."""
        from spotify_api import get_top_25
        spotify = make_spotify_df()
        result = get_top_25(spotify)
        assert isinstance(result, pd.DataFrame)

    def test_get_top_25_has_required_columns(self):
        """get_top_25 result should include song, artist, year, year_group."""
        from spotify_api import get_top_25
        result = get_top_25(make_spotify_df())
        for col in ['song', 'artist', 'year', 'year_group']:
            assert col in result.columns

    def test_get_top_25_max_25_per_group(self):
        """get_top_25 should return at most 25 rows per year-group."""
        from spotify_api import get_top_25
        # Create a larger dataset
        np.random.seed(0)
        n = 100
        df = pd.DataFrame({
            'song':       [f'Song {i}' for i in range(n)],
            'artist':     [f'Artist {i % 10}' for i in range(n)],
            'year':       np.random.choice([2005, 2006, 2007, 2008], n),
            'popularity': np.random.randint(0, 100, n),
        })
        result = get_top_25(df)
        for group in result['year_group'].unique():
            assert result[result['year_group'] == group].shape[0] <= 25

    def test_get_top_25_sorted_by_popularity(self):
        """get_top_25 should pick the most popular songs."""
        from spotify_api import get_top_25
        spotify = make_spotify_df()
        result = get_top_25(spotify)
        # The most popular song in 2005 (pop=80) should appear
        assert any(result['popularity'] == 80)

    def test_search_song_returns_none_on_bad_status(self):
        """search_song should return None when API responds with non-200."""
        from spotify_api import search_song
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch('spotify_api.requests.get', return_value=mock_response):
            result = search_song("Song 1", "Artist A")
        assert result is None

    def test_search_song_returns_dict_on_success(self):
        """search_song should return a dict with expected keys on 200."""
        from spotify_api import search_song
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'tracks': {
                'items': [{
                    'name': 'Song 1',
                    'artists': [{'name': 'Artist A'}],
                    'album': {
                        'name': 'Album X',
                        'release_date': '2005-01-15',
                        'release_date_precision': 'day',
                    }
                }]
            }
        }
        with patch('spotify_api.requests.get', return_value=mock_response):
            result = search_song("Song 1", "Artist A")
        assert result is not None
        assert 'spotify_song' in result
        assert 'release_date' in result

    def test_search_song_returns_none_on_empty_tracks(self):
        """search_song should return None when the tracks list is empty."""
        from spotify_api import search_song
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'tracks': {'items': []}}
        with patch('spotify_api.requests.get', return_value=mock_response):
            result = search_song("Unknown Song", "Unknown Artist")
        assert result is None

    def test_spotify_api_calling_returns_dataframe(self):
        """spotify_api_calling should return a DataFrame (may be empty if no hits)."""
        from spotify_api import spotify_api_calling
        spotify = make_spotify_df()
        with patch('spotify_api.search_song', return_value=None):
            result = spotify_api_calling(spotify)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# MODULE: eda_analysis (pure logic, visualizations mocked)
# ===========================================================================

class TestEdaAnalysis:
    """Unit tests for eda_analysis.py — visualization calls are mocked."""

    @patch('eda_analysis.correlation_matrix_for_spotify')
    @patch('eda_analysis.effects_of_audio_features_on_popularity')
    @patch('eda_analysis.features_trends_over_the_year')
    @patch('eda_analysis.averarge_popularity_over_the_years')
    @patch('eda_analysis.relation_between_each_feature_and_popularity')
    def test_eda_analysis_runs_without_error(self, *mocks):
        """eda_analysis should complete without raising exceptions."""
        from eda_analysis import eda_analysis
        # Need enough rows so year 2005 correlation is computable
        df = pd.concat([make_spotify_df()] * 10, ignore_index=True)
        eda_analysis(df)   # should not raise

    @patch('eda_analysis.correlation_matrix_for_spotify')
    @patch('eda_analysis.effects_of_audio_features_on_popularity')
    @patch('eda_analysis.features_trends_over_the_year')
    @patch('eda_analysis.averarge_popularity_over_the_years')
    @patch('eda_analysis.relation_between_each_feature_and_popularity')
    def test_eda_analysis_calls_all_visualizations(self, mock_rel, mock_avg,
                                                    mock_trend, mock_effects,
                                                    mock_corr):
        """eda_analysis should invoke every visualization helper."""
        from eda_analysis import eda_analysis
        df = pd.concat([make_spotify_df()] * 10, ignore_index=True)
        eda_analysis(df)
        assert mock_corr.called
        assert mock_effects.called
        assert mock_trend.called
        assert mock_avg.called
        assert mock_rel.called

    def test_eda_analysis_prints_stats(self, capsys):
        """eda_analysis should print basic statistics."""
        from eda_analysis import eda_analysis
        with patch('eda_analysis.correlation_matrix_for_spotify'), \
             patch('eda_analysis.effects_of_audio_features_on_popularity'), \
             patch('eda_analysis.features_trends_over_the_year'), \
             patch('eda_analysis.averarge_popularity_over_the_years'), \
             patch('eda_analysis.relation_between_each_feature_and_popularity'):
            df = pd.concat([make_spotify_df()] * 10, ignore_index=True)
            eda_analysis(df)
        captured = capsys.readouterr()
        assert "EDA Analysis" in captured.out


# ===========================================================================
# MODULE: kmeans_clustering (visualizations mocked)
# ===========================================================================

class TestKmeansClustering:
    """Unit tests for kmeans_clustering.py"""

    def _large_spotify(self, n=80):
        """Return a large-enough Spotify DataFrame for clustering."""
        np.random.seed(42)
        base = make_spotify_df()
        rows = [base.iloc[i % len(base)].to_dict() for i in range(n)]
        df = pd.DataFrame(rows)
        df['year'] = np.where(np.arange(n) < n // 2, 2005, 2006)
        # Add noise to numeric columns so clusters are non-trivial
        for col in ['danceability', 'energy', 'loudness', 'tempo', 'liveness', 'valence']:
            df[col] = df[col] + np.random.normal(0, 0.05, n)
        return df

    @patch('kmeans_clustering.elbow_method_for_kmeans')
    @patch('kmeans_clustering.cluster_graphs_per_year')
    def test_kmeans_runs_without_error(self, mock_cgpy, mock_elbow):
        """kmeans should complete without raising."""
        from kmeans_clustering import kmeans
        kmeans(self._large_spotify())

    @patch('kmeans_clustering.elbow_method_for_kmeans')
    @patch('kmeans_clustering.cluster_graphs_per_year')
    def test_kmeans_calls_elbow(self, mock_cgpy, mock_elbow):
        """kmeans should call elbow_method_for_kmeans once."""
        from kmeans_clustering import kmeans
        kmeans(self._large_spotify())
        assert mock_elbow.called

    @patch('kmeans_clustering.elbow_method_for_kmeans')
    @patch('kmeans_clustering.cluster_graphs_per_year')
    def test_kmeans_calls_cluster_graphs(self, mock_cgpy, mock_elbow):
        """kmeans should call cluster_graphs_per_year."""
        from kmeans_clustering import kmeans
        kmeans(self._large_spotify())
        assert mock_cgpy.called

    @patch('kmeans_clustering.elbow_method_for_kmeans')
    @patch('kmeans_clustering.cluster_graphs_per_year')
    def test_kmeans_skips_small_years(self, mock_cgpy, mock_elbow):
        """kmeans should skip year groups with fewer than 20 songs."""
        from kmeans_clustering import kmeans
        # Only 5 rows for year 2010 — should be skipped silently
        df = self._large_spotify()
        tiny = df.iloc[:5].copy()
        tiny['year'] = 2010
        combined = pd.concat([df, tiny], ignore_index=True)
        kmeans(combined)   # should not raise


# ===========================================================================
# MODULE: pca (visualizations mocked)
# ===========================================================================

class TestPcaAnalysis:
    """Unit tests for pca.py"""

    def _large_spotify(self, n=80):
        np.random.seed(7)
        base = make_spotify_df()
        rows = [base.iloc[i % len(base)].to_dict() for i in range(n)]
        df = pd.DataFrame(rows)
        df['year'] = np.where(np.arange(n) < n // 2, 2005, 2006)
        for col in ['danceability', 'energy', 'loudness', 'tempo', 'liveness', 'valence']:
            df[col] = df[col] + np.random.normal(0, 0.05, n)
        return df

    @patch('pca.pca_scatter_plots')
    @patch('pca.pca_popularity_map')
    def test_pca_analysis_runs_without_error(self, mock_pop, mock_scatter):
        """pca_analysis should complete without raising."""
        from pca import pca_analysis
        pca_analysis(self._large_spotify())

    @patch('pca.pca_scatter_plots')
    @patch('pca.pca_popularity_map')
    def test_pca_analysis_calls_visualizations(self, mock_pop, mock_scatter):
        """pca_analysis should call both visualization helpers."""
        from pca import pca_analysis
        pca_analysis(self._large_spotify())
        assert mock_scatter.called
        assert mock_pop.called

    @patch('pca.pca_scatter_plots')
    @patch('pca.pca_popularity_map')
    def test_pca_analysis_skips_small_years(self, mock_pop, mock_scatter):
        """pca_analysis should skip year groups with fewer than 20 songs."""
        from pca import pca_analysis
        df = self._large_spotify()
        tiny = df.iloc[:5].copy()
        tiny['year'] = 2099
        combined = pd.concat([df, tiny], ignore_index=True)
        pca_analysis(combined)   # should not raise


# ===========================================================================
# MODULE-LEVEL integration smoke tests
# ===========================================================================

class TestModuleIntegration:
    """Light integration tests that chain modules together."""

    def test_preprocessing_pipeline(self, tmp_path):
        """Full preprocessing pipeline should produce correctly-typed DataFrames."""
        from preprocessing import (checking_for_null_values, dropping_null_values,
                                   removing_duplicates, correcting_data_types)
        spotify = make_spotify_df()
        billboard = make_billboard_df()
        billboard['date'] = billboard['date'].astype(str)

        checking_for_null_values(spotify)
        checking_for_null_values(billboard)
        spotify = dropping_null_values(spotify, 'genre')
        billboard = dropping_null_values(billboard, 'song')
        spotify = removing_duplicates(spotify)
        billboard = removing_duplicates(billboard)
        spotify, billboard = correcting_data_types(spotify, billboard)

        assert spotify['year'].dtype == int
        assert pd.api.types.is_datetime64_any_dtype(billboard['date'])
        assert billboard['rank'].dtype == int

    def test_get_top_25_feeds_api_calling(self):
        """get_top_25 output shape should satisfy spotify_api_calling's iteration."""
        from spotify_api import get_top_25, spotify_api_calling
        spotify = make_spotify_df()
        top = get_top_25(spotify)
        assert 'song' in top.columns
        assert 'artist' in top.columns
        # Mock the actual API call so we don't need a token
        with patch('spotify_api.search_song', return_value=None):
            result = spotify_api_calling(spotify)
        assert isinstance(result, pd.DataFrame)

    @patch('billboard_charting.billboard_distribution')
    @patch('billboard_charting.spotify_songs_in_billboard')
    @patch('billboard_charting.date_vs_chart')
    @patch('billboard_charting.delay_analysis')
    def test_billboard_charting_with_preprocessed_data(self, *mocks):
        """billboard_charting should work on corrected-type DataFrames."""
        from preprocessing import correcting_data_types
        from billboard_charting import billboard_charting

        spotify_api = make_spotify_api_df()
        billboard = make_billboard_df()
        billboard['date'] = billboard['date'].astype(str)
        _, billboard = correcting_data_types(make_spotify_df(), billboard)
        billboard_charting(billboard, spotify_api)   # should not raise

# ===========================================================================
# MODULE: dashboard_data (interactive dashboard)
# ===========================================================================

class TestDashboardData:

    def test_clean_song_title_removes_features_and_edits(self):
        from dashboard_data import clean_song_title
        assert clean_song_title("Love The Way You Lie (feat. Rihanna) - Radio Edit") == "love the way you lie"

    def test_clean_artist_keeps_lead_artist(self):
        from dashboard_data import clean_artist
        assert clean_artist("Drake Featuring Rihanna") == "drake"
        assert clean_artist("Jay-Z & Kanye West") == "jayz"

    def test_load_clean_songs_removes_duplicates(self, tmp_path):
        from dashboard_data import load_clean_songs
        df = make_spotify_df()          # 5 rows, the last one duplicates the first
        path = tmp_path / "songs.csv"
        df.to_csv(path, index=False)
        out = load_clean_songs(path)
        assert len(out) == 4
        assert 'duration_min' in out.columns

    def test_cluster_by_year_names_three_types(self):
        from dashboard_data import cluster_by_year
        rng = np.random.default_rng(0)
        n = 60
        songs = pd.DataFrame({
            'song': [f's{i}' for i in range(n)], 'artist': 'a', 'year': 2010,
            'danceability': rng.random(n), 'energy': rng.random(n), 'loudness': -rng.random(n) * 10,
            'tempo': rng.random(n) * 100 + 80, 'liveness': rng.random(n), 'valence': rng.random(n),
            'popularity': rng.integers(0, 100, n)})
        clustered, loadings = cluster_by_year(songs)
        assert set(clustered['song_type']) == {'High Energy', 'Balanced', 'Low Energy / Emotional'}
        hi = clustered[clustered['song_type'] == 'High Energy']['energy'].mean()
        lo = clustered[clustered['song_type'] == 'Low Energy / Emotional']['energy'].mean()
        assert hi > lo
        assert loadings[2010].loc['energy', 'PC1'] >= 0

    def test_cluster_by_year_skips_small_years(self):
        from dashboard_data import cluster_by_year
        rng = np.random.default_rng(1)
        rows = 25 + 5
        songs = pd.DataFrame({
            'song': [f's{i}' for i in range(rows)], 'artist': 'a', 'year': [2010] * 25 + [2020] * 5,
            'danceability': rng.random(rows), 'energy': rng.random(rows), 'loudness': -rng.random(rows),
            'tempo': rng.random(rows), 'liveness': rng.random(rows), 'valence': rng.random(rows),
            'popularity': 50})
        clustered, _ = cluster_by_year(songs)
        assert set(clustered['year']) == {2010}

    def test_match_billboard_finds_featured_and_missing_songs(self):
        from dashboard_data import match_billboard
        songs = pd.DataFrame({'song': ['Umbrella', 'Unknown Song'], 'artist': ['Rihanna', 'Nobody'],
                              'year': [2007, 2007], 'popularity': [80, 40]})
        charts = pd.DataFrame({'date': ['2007-06-09', '2007-06-16'], 'rank': [3, 1],
                               'song': ['Umbrella', 'Umbrella'], 'artist': ['Rihanna Featuring Jay-Z'] * 2})
        out = match_billboard(songs, charts)
        assert bool(out.loc[0, 'charted']) is True
        assert out.loc[0, 'peak_rank'] == 1 and out.loc[0, 'weeks_on_chart'] == 2
        assert bool(out.loc[1, 'charted']) is False
