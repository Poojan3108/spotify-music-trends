# 🎵 Two Decades of Hit Music: Spotify & Billboard Trend Analysis

How did the sound of popular music change from 1999 to 2019, what makes a hit popular, and do Spotify hits also succeed on the Billboard Hot 100? This project combines **Spotify audio features, the Spotify Web API, and Billboard Hot 100 chart history** to answer those questions, using exploratory analysis, K-Means clustering, and PCA, presented in an interactive dashboard.

**Live dashboard:** [add your Streamlit link here]

*Originally built as my final project for DSCI 510 (Principles of Programming for Data Science) at USC, then extended with an interactive dashboard, corrected data cleaning, and more accurate dataset matching.*

## Key findings

**1. Hit songs got shorter and less intense.**
- Average song length fell from **4.2 minutes (1999) to 3.3 minutes (2019)**, about 57 seconds shorter.
- Energy peaked in **2010** (0.78) and fell to 0.66 by 2019, while danceability (0.63 → 0.72) and acousticness (0.10 → 0.19) rose sharply after 2013.

**2. No single audio feature predicts popularity.**
Every feature's correlation with Spotify popularity is between -0.05 and 0.05. Popularity depends on a combination of factors, including many that audio can't capture (marketing, artist fame, culture).

**3. Hits fall into three types, and all three are equally popular.**
Year-by-year K-Means clustering (k=3, chosen with the elbow method) on 6 audio features finds **High Energy**, **Balanced**, and **Low Energy / Emotional** songs. PCA compresses the 6 features into two interpretable dimensions:
- **PC1 – Intensity:** driven by energy and loudness
- **PC2 – Emotion / rhythm:** driven by valence and danceability

Average popularity is nearly identical across the three types (59–60), so hits come from every style.

**4. Most Spotify hits also chart in the US, and most reach the top 10.**
- **75%** of the 1,926 songs appeared on the Billboard Hot 100 (1,450 songs).
- Of those, **56%** reached the top 10 and **209** hit #1. The median song spent **23 weeks** on the chart.
- Songs that charted average a Spotify popularity of 61.4 vs 54.3 for those that didn't. Many non-charting songs are UK and European hits, which Billboard (a US chart) doesn't measure.

## Corrections in this version

Re-examining the original pipeline uncovered issues that affected the results:

| Issue | Effect | Fix |
|---|---|---|
| `removing_duplicates()` result was never saved in `main.py` | 59 duplicate songs stayed in the analysis | Assign the cleaned result back (plus remove 15 re-listed songs with the same title and artist) |
| The "Featuring" cleaning pattern expected `Featuring.` with a period | Matched **0 of 32,022** Billboard rows with featured artists | Corrected pattern: `"Drake Featuring Rihanna"` → `"Drake"` |
| Billboard matching used a 90-day window after the release date returned by the Spotify API | The API often returns a later compilation or remaster date, so songs that charted *before* that date were counted as never charting. The original analysis reported **57%** of top songs never charting; with corrected matching, it is **11.5%** for the same songs. | Match every song on cleaned title + lead artist across all chart weeks since 1998 |
| Dropping Billboard rows with an empty `last-week` | Those rows are each song's **first week** on the chart | Kept them |

## Interactive dashboard

| Tab | What you can do |
|---|---|
| 📈 Trends over time | Compare how any audio features changed from 1999 to 2019 |
| 🎯 What drives popularity? | See each feature's correlation with popularity and explore scatter plots |
| 🎧 Song types | Pick a year to see its songs clustered and mapped with PCA, with each cluster's profile |
| 📊 Spotify vs Billboard | Chart rates, peak ranks, and a searchable table of every song |
| 🔍 Song explorer | Compare any song with the average hit from its year |

## Data sources

| # | Source | Type | Fields | Size |
|---|---|---|---|---|
| 1 | [Top Hits Spotify 2000–2019](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019/data) | CSV | artist, song, year, popularity, danceability, energy, loudness, tempo, valence, genre, ... | 2,000 songs |
| 2 | [Spotify Web API](https://developer.spotify.com/documentation/web-api) | API | album, release date | top songs per period |
| 3 | [Billboard Hot 100](https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs) | CSV | date, rank, song, artist, weeks on chart | 330,087 weekly entries |

## Project structure

```
├── app.py                      # Streamlit dashboard
├── data/
│   ├── songs_normalize.csv     # Spotify top hits dataset
│   └── billboard_matches.csv   # Billboard results for each song (built by build_dashboard_data.py)
├── src/
│   ├── main.py                 # original analysis pipeline (EDA, K-Means, PCA, API, Billboard)
│   ├── data_loader.py, preprocessing.py, eda_analysis.py, kmeans_clustering.py,
│   │   pca.py, spotify_api.py, billboard_charting.py, visualization.py
│   ├── dashboard_data.py       # data preparation for the dashboard
│   ├── build_dashboard_data.py # builds data/billboard_matches.csv from the full Billboard file
│   └── tests.py                # 52 pytest tests
└── doc/                        # project report and presentation
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Original analysis pipeline:** download the Billboard dataset into `data/charts.csv`, create an empty `results/` folder, add `SPOTIFY_API_TOKEN=...` to a `.env` file, then:
```bash
cd src
python main.py
```

**Rebuild the dashboard's Billboard data:** `cd src && python build_dashboard_data.py`

**Tests:** `cd src && python -m pytest tests.py`

---
*Note: AI tools (Claude, ChatGPT) were used to help refine code structure, build the dashboard, and fix bugs. The core analysis design and logic are my own.*
