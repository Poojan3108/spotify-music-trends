# Preprocessing the data before analysis

# Done based on the insights visible from the raw data

import pandas as pd

# Checking for null values
def checking_for_null_values(df):
    null_counts = df.isnull().sum()
    print("Null values in each column:")
    print(null_counts)

# Dropping the null values
def dropping_null_values(df, feauture):
    df = df.dropna(subset=[feauture])
    return df

# Removing duplicates
def removing_duplicates(df):
    df = df.drop_duplicates()
    return df

# Correcting the data types
def correcting_data_types(spotify_df, billboard_df):
    spotify_df['year'] = spotify_df['year'].astype(int)
    billboard_df['date'] = pd.to_datetime(billboard_df['date'])
    billboard_df['rank'] = billboard_df['rank'].astype(int)
    # Keep only the lead artist: "Drake Featuring Rihanna" -> "Drake"
    billboard_df['artist'] = billboard_df['artist'].str.replace(r'\s+Featuring\b.*', '', regex=True).str.strip()
    return spotify_df, billboard_df
