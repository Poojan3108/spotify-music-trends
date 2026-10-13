# Calling the Apotify API to get the metedata about the songs.

import pandas as pd
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Data to be called through the API
def spotify_api_calling(spotify):
    spotify_data = []
    top_songs_df = get_top_25(spotify)
    for _, row in top_songs_df.iterrows():

        result = search_song(row['song'], row['artist'])

        if result:
            result.update({
                "original_song": row['song'],
                "original_artist": row['artist'],
                "year_group": row['year_group']
            })

            spotify_data.append(result)

    spotify_api_df = pd.DataFrame(spotify_data)
    return spotify_api_df

# Getting the top 25 over the span of 2 years
def get_top_25(spotify):
    top_songs_2yr = []

    years = sorted(spotify['year'].unique())

    for i in range(0, len(years), 2):
        year_group = years[i:i+2]

        data = spotify[spotify['year'].isin(year_group)]

        top25 = data.sort_values(by='popularity', ascending=False).head(25)

        top25 = top25[['song', 'artist', 'year', 'popularity']]

        top25['year_group'] = f"{year_group[0]}-{year_group[-1]}"

        top_songs_2yr.append(top25)

    top_songs_df = pd.concat(top_songs_2yr).reset_index(drop=True)

    return top_songs_df

# Making the API call
def search_song(song, artist):
    token = os.getenv("SPOTIFY_API_TOKEN")
    url = "https://api.spotify.com/v1/search"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    query = f"track:{song} artist:{artist}"

    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        return None

    data = response.json()

    try:
        track = data['tracks']['items'][0]

        release_date = track['album']['release_date']
        release_precision = track['album']['release_date_precision']

        return {
            "spotify_song": track['name'],
            "spotify_artist": track['artists'][0]['name'],
            "album": track['album']['name'],
            "release_date": release_date,
            "release_date_precision": release_precision   # year / month / day
        }

    except:
        return None
