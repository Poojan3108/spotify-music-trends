# EDA Analysis for finding the best relevant features and relations between the features

# In this section of the code I have used AI to refine my Eda Analysis code

from sklearn.preprocessing import StandardScaler
from visualization import correlation_matrix_for_spotify, effects_of_audio_features_on_popularity, features_trends_over_the_year, averarge_popularity_over_the_years, relation_between_each_feature_and_popularity

def eda_analysis(spotify_df):
    
    print("\nEDA Analysis of Spotify Dataset:\n")

    # Selecting the most relevant features for EDA
    features = ['artist', 'song', 'duration_ms', 'explicit', 'year', 'popularity',
       'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'genre']
    
    yearly_correlations = {}
    year_datasets = {}

    spotify = spotify_df[features].copy()
    
    # Checking the basic stats
    print("\nChecking the basic statistics of the Spotify dataset:\n")
    print(spotify.describe())

    # Checking the yearly average
    print("\nChecking the yearly average values:\n")
    yearly_avg = spotify.groupby('year').mean(numeric_only=True)
    print(yearly_avg)

    # Correlation analysis of features with popularity
    print("\nCorrelation of features with popularity:\n")
    correlation_matrix = spotify.corr(numeric_only=True)
    print(correlation_matrix['popularity'].sort_values(ascending=False))

    # Year-wise correlation analysis for the features
    print("\nCorrelation of features with popularity by year:\n")

    for year in spotify['year'].unique():
        temp = spotify[spotify['year'] == year]
        corr = temp.corr(numeric_only=True)['popularity']
        yearly_correlations[year] = corr

    print("For the year 2005:\n")
    print(yearly_correlations[2005])

    # Preparing features for clustering analysis
    cluster_features = [
    'danceability', 'energy', 'loudness',
    'tempo', 'liveness', 'valence'
    ]

    X = spotify[cluster_features]
    spotify = spotify[ cluster_features + ['artist'] + ['song'] + ['popularity'] + ['year']]    

    # Normalizing the features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Year-wise data splitting
    for year in spotify['year'].unique():
        year_datasets[year] = spotify[spotify['year'] == year]

    print("\nCorrelation Matrix\n")
    correlation_matrix_for_spotify(spotify)

    print("\nEffects of Audio Features on Popularity\n")
    effects_of_audio_features_on_popularity(spotify)

    print("\nTrends of Audio Features Over the Years\n")
    features_trends_over_the_year(spotify)

    print("\nAverage Popularity Over the Years\n")
    averarge_popularity_over_the_years(spotify, yearly_avg)

    print("\nRelation Between Each Feature and Popularity\n")
    relation_between_each_feature_and_popularity(spotify)
