# Run once to create data/billboard_matches.csv from the large Billboard file (data/charts.csv).
# The dashboard then only needs the two small files in data/.
#   cd src && python build_dashboard_data.py

from pathlib import Path

import pandas as pd

from dashboard_data import load_clean_songs, match_billboard

DATA = Path(__file__).resolve().parent.parent / "data"

songs = load_clean_songs(DATA / "songs_normalize.csv")
charts = pd.read_csv(DATA / "charts.csv")
matches = match_billboard(songs, charts)
matches.to_csv(DATA / "billboard_matches.csv", index=False)
print(f"{len(songs)} songs, {matches['charted'].mean():.1%} charted on the Billboard Hot 100")
print(f"Saved {DATA / 'billboard_matches.csv'}")
