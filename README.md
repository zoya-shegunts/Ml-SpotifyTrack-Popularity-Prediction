# Spotify Track Popularity: EDA & Regression

Exploratory data analysis and regression modelling on Spotify track data. The goal is to find out how well a track's **popularity score (0–100)** can be predicted from its audio features and metadata.

> This repository contains the **EDA** and **regression** parts of a larger group machine learning project (which also included classification and unsupervised learning, done by teammates).

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Part 1: Exploratory Data Analysis](#part-1-exploratory-data-analysis)
- [Part 2: Regression](#part-2-regression)
- [Key Takeaways](#key-takeaways)
- [Limitations & Notes](#limitations--notes)

---

## Dataset

Two CSV files of Spotify tracks collected from playlists via the Spotify API:

| File | Rows | Popularity range | Mean popularity |
|---|---|---|---|
| `high_popularity_spotify_data.csv` | 1,686 | 68 – 100 | 75.8 |
| `low_popularity_spotify_data.csv` | 3,145 | 11 – 68 | 43.5 |

Each file has 29 columns. Merged, they give **4,831 tracks across 35 playlist genres**.

**Main columns**

- **Audio features:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `key`, `mode`, `time_signature`
- **Metadata:** `track_name`, `track_artist`, `track_album_name`, `track_album_release_date`, `playlist_name`, `playlist_genre`, `playlist_subgenre`
- **Target:** `track_popularity` (0–100)

---

## Project Structure

```
.
├── high_popularity_spotify_data.csv
├── low_popularity_spotify_data.csv
├── ML_Project_EDA.ipynb          # Data loading, merging, cleaning, EDA
└── ML_Project_Regression.ipynb   # Feature engineering + regression models
```

---

## Part 1: Exploratory Data Analysis

`ML_Project_EDA.ipynb`

**Data preparation**
- Loaded both CSVs and concatenated them into one dataframe.
- Added a binary column `pop_class` (`1` = from the high-popularity file, `0` = from the low-popularity file). About 65% of tracks are class 0 and 35% class 1.
- Dropped ID and link columns with no predictive value: `track_href`, `uri`, `analysis_url`, `id`, `track_album_id`, `playlist_id`, `track_id`.
- Checked for duplicate rows: **none found**.

**Analysis performed**
- Histograms of all numerical features
- Correlation heatmap
- Distribution of `track_popularity` (mean ≈ 54.8, std ≈ 19.8, range 11–100)
- Pairplots of each feature against popularity
- KDE plots and boxplots of each feature split by `pop_class`
- Boxplots of energy, tempo, duration and acousticness by genre

**Findings**
- Features have very different scales and shapes: `duration_ms` is right-skewed, and `acousticness` and `instrumentalness` are very heavily right-skewed. This motivates scaling and skew correction before modelling.
- Energy, loudness and danceability are positively correlated with each other; acousticness is strongly negatively correlated with energy and loudness.
- No audio feature has a strong linear correlation (|r| > 0.5) with popularity. The only one that does is `pop_class` (r ≈ 0.78), which is derived from the popularity split itself.
- Popular tracks are on average more energetic (0.67 vs. 0.54), louder (−6.7 dB vs. −10.7 dB) and slightly more danceable (0.65 vs. 0.61) than less popular ones.
- Some genres are acoustically distinct (classical, ambient, rock) while others overlap heavily (pop / k-pop, latin / afrobeats, arabic / turkish).

---

## Part 2: Regression

`ML_Project_Regression.ipynb`

**Goal:** predict the exact `track_popularity` score.

### Preprocessing & feature engineering

1. **Missing values:** numeric columns filled with the median; rows with missing `mode` or `track_album_name` dropped, leaving 4,830 rows.
2. **Encoding:**
   - One-hot encoding of `playlist_genre`, `playlist_subgenre` and `type`
   - Label encoding of `playlist_name`
   - `track_artist`: top 20 artists kept, all others grouped as `Other`, then label-encoded
3. **Skew correction:** features with skew > 0.7 get `log1p`; features with skew < −0.7 are reflected (max − x) and then `log1p`.
4. **Scaling:** `StandardScaler`.
5. **Split:** 70% train / 30% test (`random_state=42`).

**Experiments that were tried and rejected**
- Merging `energy` and `loudness` into a single ratio feature to reduce multicollinearity: R² went down, so the features were kept separate.
- Cross-validation: gave lower R² than the single train/test split, so the split was kept for evaluation.

### Models and results

| Model | Test R² | Test MSE | Test RMSE |
|---|---|---|---|
| Linear Regression | 0.100 | 357.55 | 18.91 |
| Polynomial Regression (best: degree 2) | 0.193 | 320.49 | 17.90 |
| Decision Tree | 0.351 | 257.96 | 16.06 |
| **Random Forest (200 trees)** | **0.655** | **136.98** | **11.70** |
| Random Forest + Lasso-selected features | 0.655 | 136.98 | 11.70 |

**Polynomial regression by degree**

| Degree | Train R² | Test R² |
|---|---|---|
| 1 | 0.121 | 0.100 |
| 2 | 0.244 | 0.193 |
| 3 | 0.432 | −0.569 |
| 5 | 1.000 | −504 |

Degrees 3 and 5 overfit severely, so degree 2 is the best polynomial model.

**Random Forest:** train R² 0.951 vs. test R² 0.655, so it also fits the training data much more closely than the test data. A feature-importance plot is included in the notebook.

**Lasso feature selection:** `LassoCV` (5-fold) keeps 15 features. The strongest coefficients are `loudness` (−4.89), `energy` (−3.59), `instrumentalness` (−2.93), `acousticness` (−2.74), `playlist_name` (+1.56) and `track_artist` (+1.17). A Random Forest trained on only these features matches the full-feature model (R² 0.655).

---

## Key Takeaways

- **Linear models do poorly** (R² ≈ 0.10), so audio features have no simple linear relationship with popularity.
- **Non-linear models help a lot.** Random Forest roughly halves the RMSE of linear regression (11.7 vs. 18.9 popularity points).
- **High-degree polynomials overfit** badly; model complexity has to be controlled.
- Popularity is influenced by many things that audio features don't capture (artist reach, marketing, playlist placement, timing).

---

## Limitations & Notes

- **Possible information leakage.** `playlist_name` and `track_artist` are used as features and are among Lasso's top-selected variables. Because the high- and low-popularity tracks came from different playlists, `playlist_name` may indirectly reveal which popularity group a track belongs to. The Random Forest R² of 0.655 should be interpreted with this in mind; a useful follow-up is to re-run the models without these two columns.
- **Large train/test gap** for Random Forest (0.951 vs. 0.655) suggests overfitting; hyperparameter tuning (e.g. limiting depth) is not yet explored.
- **Single split.** Results come from one random split (`random_state=42`) and may vary slightly with a different split.
- **Folder layout matters.** The regression notebook imports `ML_Project_EDA` by name and the EDA notebook reads the CSVs by relative path.

---

## Authors

Add your name here.

## License

Add a license of your choice (e.g. MIT).
