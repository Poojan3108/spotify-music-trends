# PCA Analysis

# In this section of the code I have used AI to generate the code for PCA Analysis that integrates with my existing code.

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from visualization import pca_scatter_plots, pca_popularity_map

# Main function for PCA Analysis
def pca_analysis(spotify):
    
    cluster_features = [
    'danceability', 'energy', 'loudness',
    'tempo', 'liveness', 'valence'
    ]

    year_clusters = {}
    year_pca_data = {}

    for year in sorted(spotify['year'].unique()):

        data = spotify[spotify['year'] == year]

        if len(data) < 20:
            continue

        X = data[cluster_features]

        # Scaling the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA function
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        data = data.copy()
        data['cluster'] = clusters
        data['PC1'] = X_pca[:, 0]
        data['PC2'] = X_pca[:, 1]

        year_clusters[year] = data
        year_pca_data[year] = (pca, X_pca)

    pca_scatter_plots(year_clusters)
    pca_popularity_map(year_pca_data, year_clusters)
