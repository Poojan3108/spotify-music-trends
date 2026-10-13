# Comparing the Spotify and Billboard data

import pandas as pd
import re
from visualization import billboard_distribution, spotify_songs_in_billboard, date_vs_chart, delay_analysis

def billboard_charting(billboard, spotify_api_df):
    spotify_api_df['song_clean'] = spotify_api_df['spotify_song'].apply(clean_text)
    spotify_api_df['artist_clean'] = spotify_api_df['spotify_artist'].apply(clean_text)

    billboard['song_clean'] = billboard['song'].apply(clean_text)
    billboard['artist_clean'] = billboard['artist'].apply(clean_text)

    spotify_api_df['release_date'] = pd.to_datetime(spotify_api_df['release_date'], errors='coerce')
    billboard['date'] = pd.to_datetime(billboard['date'], errors='coerce')

    matched_results = []

    for _, s_row in spotify_api_df.iterrows():

        song = s_row['song_clean']
        artist = s_row['artist_clean']
        release_date = s_row['release_date']

        if pd.isna(release_date):
            continue

        end_date = release_date + pd.Timedelta(days=90)

        bb_filtered = billboard[
            (billboard['song_clean'] == song) &
            (billboard['artist_clean'].str.contains(artist)) &
            (billboard['date'] >= release_date) &
            (billboard['date'] <= end_date)
        ]

        if not bb_filtered.empty:
            best_rank = bb_filtered['rank'].min()  

            best_entry = bb_filtered.loc[bb_filtered['rank'].idxmin()]

            matched_results.append({
                "song": s_row['spotify_song'],
                "artist": s_row['spotify_artist'],
                "release_date": release_date,
                "best_rank": best_rank,
                "rank_date": best_entry['date']
            })
        
    billboard_match_df = pd.DataFrame(matched_results)
    billboard_distribution(billboard_match_df)
    spotify_songs_in_billboard(spotify_api_df, billboard_match_df)
    date_vs_chart(billboard_match_df)
    delay_analysis(billboard_match_df)
    

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text.strip()
