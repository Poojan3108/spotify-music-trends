# Loading and Handling the datasets

import pandas as pd


# Loading the Spotify Dataset
def load_spotify_data(path):
    spotify_df = pd.read_csv(path)
    return spotify_df


# Loading the Billboard Dataset
def load_billboard_data(path):
    billboard_df = pd.read_csv(path)
    return billboard_df


# Checking the data
def preview_data(df, name="Dataset"):
    print(f"\n{name} Preview:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())


# Handling Missing Values
def check_missing_values(df, name="Dataset"):
    print(f"\nMissing Values in {name}:")
    print(df.isnull().sum())


# About the data
def dataset_info(df, name="Dataset"):
    print(f"\n{name} Info:")
    print(df.info())
