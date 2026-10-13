# The main file that runs the entire data analysis pipeline for the Spotify and Billboard datasets.
# It includes data loading, preprocessing, exploratory data analysis, clustering, PCA analysis, Spotify API calls and Billboard charting analysis.

# For this project, I used AI tools to help improve the efficiency and structure of my code.
# AI was used selectively to refine certain parts of the code.
# All instances where AI was used are clearly mentioned in the code comments, specifying where and how it contributed.

from data_loader import load_billboard_data, load_spotify_data
from preprocessing import checking_for_null_values, dropping_null_values, removing_duplicates, correcting_data_types
from eda_analysis import eda_analysis
from kmeans_clustering import kmeans
from pca import pca_analysis
from spotify_api import spotify_api_calling
from billboard_charting import billboard_charting

spotify_df = load_spotify_data("../data/songs_normalize.csv")
billboard_df = load_billboard_data("../data/charts.csv")

# Importing the data
print(spotify_df.head())
print(billboard_df.head())
print("Spotify shape:", spotify_df.shape)
print("Billboard shape:", billboard_df.shape)

# Preprocessing

# Removing the null values
print(spotify_df.columns)
print(billboard_df.columns)

checking_for_null_values(spotify_df)
checking_for_null_values(billboard_df)

# Note: Billboard rows with an empty "last-week" are a song's FIRST week on the chart,
# so they are kept (dropping them would remove every song's debut week).

# Removing duplicates (the result must be assigned back, or nothing changes)
spotify_df = removing_duplicates(spotify_df)
billboard_df = removing_duplicates(billboard_df)

print(billboard_df.shape)
print(spotify_df.shape)


# Correcting the data types
spotify_df, billboard_df = correcting_data_types(spotify_df, billboard_df)


# Data Analysis

# EDA Analysis
eda_analysis(spotify_df)

# KMeans Clustering
kmeans(spotify_df)

# PCA Analysis
pca_analysis(spotify_df)

# Spotify API Calling
spotify_api_df = spotify_api_calling(spotify_df)
print(spotify_api_df.head())

# Billboard Charting Comparison
billboard_charting(billboard_df, spotify_api_df)
