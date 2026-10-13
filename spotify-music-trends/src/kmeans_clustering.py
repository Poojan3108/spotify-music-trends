# K-Means Clustering Analysis

# In this section of the code, I have used AI to improve the efficiency and structure of the K-Means clustering analysis.

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from visualization import elbow_method_for_kmeans, cluster_graphs_per_year

# Main function for K-Means
def kmeans(spotify):
    cluster_features = [
    'danceability', 'energy', 'loudness',
    'tempo', 'liveness', 'valence'
    ]

    for col in cluster_features:
        Q1 = spotify[col].quantile(0.25)
        Q3 = spotify[col].quantile(0.75)
        IQR = Q3 - Q1

        spotify = spotify[
            (spotify[col] >= Q1 - 1.5 * IQR) &
            (spotify[col] <= Q3 + 1.5 * IQR)
        ]

    print("Year wise song counts:")
    print(spotify['year'].value_counts())

    year_clusters = {}

    # Year wise clustering
    for year in sorted(spotify['year'].unique()):

        data = spotify[spotify['year'] == year]

        if len(data) < 20:
            continue

        X = data[cluster_features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        data = data.copy()
        data['cluster'] = clusters

        year_clusters[year] = data

    # Year wise Cluster table
    print("\nCluster Analysis by Year:")
    for year in year_clusters:
        print(f"\nYear: {year}")
        print(
            year_clusters[year]
            .groupby('cluster')[cluster_features + ['popularity']]
            .mean()
        )

    for year in year_clusters:
        print(f"\nYear: {year}")
        print(
            year_clusters[year]
            .groupby('cluster')['popularity']
            .mean()
            .sort_values(ascending=False)
        )

    # Using elbow method to find the best value of k
    sample_year = list(year_clusters.keys())[0]
    data = year_clusters[sample_year]

    X = data[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    elbow_method_for_kmeans(inertia, sample_year)


    # Cluster Table
    for year in year_clusters:
        print(f"\nYear: {year}")

        cluster_summary = (
            year_clusters[year]
            .groupby('cluster')[cluster_features + ['popularity']]
            .mean()
        )

        print(cluster_summary)

    # Cluster Graphs for each year
    cluster_graphs_per_year(year_clusters)

    
