# NBA MVP Prediction

A data analysis and prediction project focused on identifying NBA Most Valuable Player (MVP) award winners using historical player statistics and MVP voting data.

---

## Overview

This project aggregates decades of NBA player performance metrics alongside MVP voting records to explore patterns in MVP selection and build toward a predictive model. Data spans from the 2001 season (Allen Iverson's MVP win) through the 2025 season (Shai Gilgeous-Alexander).

---

## Project Structure

```
NBA_MVP_Prediction/
├── get_and_clean_data.py          # Downloads/cleans historical data, pulls live 2025-26 stats via nba_api
├── mvp_full_stats.py              # Builds the MVP-winners feature set used as training data
├── model_training.py              # Trains the XGBoost pipeline and predicts 2026 MVP win shares
├── NBA_stats_including_MVP.csv    # Comprehensive player stats + MVP share (1991–2020)
├── 2001-2010 MVP Data.csv         # MVP voting records, 2001–2010
├── 2010-2021 MVP Data.csv         # MVP voting records, 2010–2021
├── 2022-2023 MVP Data.csv         # MVP voting records, 2022–2023
└── .myvenv/                       # Python virtual environment
```

---

## Data Sources

Datasets are downloaded directly from Kaggle via `kagglehub`:

| Dataset | Kaggle Identifier |
|---|---|
| NBA Player Stats (including MVP Share) | `orelsorekml/nba-player-stats-including-mvp-share` |
| NBA MVP Voting Dataset (2000–2021) | `parthdande/nba-mvp-voting-dataset-2000-2021` |

---

## Datasets

### `NBA_stats_including_MVP.csv`
- **Rows:** ~13,552 player-season records
- **Seasons:** 1991–2020 (30 seasons)
- **Unique Players:** ~2,578

Includes standard box score stats, advanced metrics, and MVP voting share per player per season.

**Key Columns:**

| Category | Columns |
|---|---|
| Player Info | `Player`, `Pos_x`, `Age_x`, `Tm`, `Year` |
| Game Logs | `G_x`, `GS`, `MP_x` |
| Scoring | `PTS`, `FG`, `FGA`, `FG%`, `3P`, `3PA`, `3P%`, `2P`, `2PA`, `2P%`, `FT`, `FTA`, `FT%`, `eFG%` |
| Rebounds | `ORB`, `DRB`, `TRB` |
| Playmaking/Defense | `AST`, `STL`, `BLK`, `TOV`, `PF` |
| Advanced Metrics | `PER`, `TS%`, `WS`, `WS/48`, `BPM`, `OBPM`, `DBPM`, `VORP`, `USG%`, `3PAr`, `FTr` |
| MVP Voting | `Pts Won`, `Pts Max`, `Share`, `W/L%` |

### MVP Voting Files
Three separate CSV files covering different eras are combined into a single DataFrame:

| File | Records | Era |
|---|---|---|
| `2001-2010 MVP Data.csv` | 152 | Allen Iverson → LeBron James |
| `2010-2021 MVP Data.csv` | 161 | LeBron James → Nikola Jokic |
| `2022-2023 MVP Data.csv` | 25 | Recent seasons |

Each file shares a consistent schema: `Rank`, `Player`, `Age`, `Tm`, `First`, `Pts Won`, `Pts Max`, `Share`, `G`, `MP`, `PTS`, `TRB`, `AST`, `STL`, `BLK`, `FG%`, `3P%`, `FT%`, `WS`, `WS/48`, `year`.

---

## How It Works

### `get_and_clean_data.py`

1. **Download datasets** from Kaggle using `kagglehub` (historical player stats + MVP voting records).
2. **Clean data** — drops duplicates and normalizes player name spellings (accented characters) across all datasets, caching cleaned copies under `cleaned_data/`.
3. **Concatenate MVP voting sets** (2001–2010, 2010–2021) into a single `mvps_pd` DataFrame, dropping cross-file duplicates.
4. **Pull live 2025-26 stats** via `nba_api` — one call for Advanced metrics (efficiency/usage), one for Base metrics (box-score counting stats) — merges them on player ID, derives `2P%` from the counting stats, rename/remap columns to match the historical schema, and re-fetch/overwrite `cleaned_data/cleaned_2026_stats.csv` on every run (unlike the historical datasets, which are cached once).
5. **Identify MVP winners** — filters `mvps_pd` down to `Rank == "1"` rows for each season.

### `mvp_full_stats.py`

Joins each historical MVP winner (by player name + season year) against `full_stats_pd` to build `mvp_stats_list` — the full advanced-stats row for every MVP winner, used as training data.

### `model_training.py`

1. Selects feature columns common to both the historical stats and the live 2025-26 stats.
2. Ordinal-encodes position (`Pos_x`) and splits the historical data into train/test sets.
3. Fits a `scikit-learn` `Pipeline` (`StandardScaler` → `XGBRegressor`) to predict MVP voting share from player statistics.
4. Filters 2025-26 players to those with at least 60 games played, then runs the fitted pipeline on them to produce **predicted MVP win shares for the 2026 season**, sorted and printed.
5. Reports mean squared error on the held-out test set.
6. Backtests the model against the last ~10 historical seasons, comparing the player the model ranks #1 by predicted share against the season's actual MVP.

---

## Output

Running `model_training.py` prints two tables.

### 2026 Top 10 (Predicted Win Share)

The `results` DataFrame, sorted by `Predicted Vote Share` descending — the model's current MVP-race ranking for the 2025-26 season, built from live in-season stats pulled via `nba_api`. Example:

| Player | Predicted Vote Share |
|---|---|
| Nikola Jokic | 0.484712 |
| Shai Gilgeous-Alexander | 0.456103 |
| Tyrese Maxey | 0.392771 |
| Kawhi Leonard | 0.342886 |
| Luka Doncic | 0.258167 |
| Donovan Mitchell | 0.169055 |
| Anthony Edwards | 0.166336 |
| Pascal Siakam | 0.094490 |
| Victor Wembanyama | 0.092492 |
| Jamal Murray | 0.083234 |

Note: `get_and_clean_data.py` re-fetches `cleaned_2026_stats.csv` from `nba_api` on every run (it's live in-season data, not a one-time historical snapshot), so this table reflects predictions as of whenever the pipeline was last run — re-run it for the latest games. Earlier in a season, with fewer games played, predicted shares can cluster near-identically across many players since there isn't enough separation in the underlying stats yet.

### Backtest (Predicted #1 vs. Actual MVP)

The `backtest_results` DataFrame. For each of the last ~10 seasons present in the historical dataset (`full_stats_pd`, which runs through 2020), the fitted pipeline predicts `Share` for every player in that season; the player with the highest predicted share (`Predicted MVP`) is compared against the player with the highest actual `Share` (`Actual MVP`).

| Season | Predicted MVP | Actual MVP | Correct |
|---|---|---|---|
| 2011 | Dwight Howard | Derrick Rose | False |
| 2012 | LeBron James | LeBron James | True |
| 2013 | LeBron James | LeBron James | True |
| 2014 | Kevin Durant | Kevin Durant | True |
| 2015 | Stephen Curry | Stephen Curry | True |
| 2016 | Stephen Curry | Stephen Curry | True |
| 2017 | Russell Westbrook | Russell Westbrook | True |
| 2018 | James Harden | James Harden | True |
| 2019 | James Harden | Giannis Antetokounmpo | False |
| 2020 | Giannis Antetokounmpo | Giannis Antetokounmpo | True |

This backtest is not a true out-of-sample test — the pipeline is trained on a random split of all seasons combined, so a backtested season's own data may have contributed to training. It's a sanity check on ranking behavior, not a rigorous walk-forward validation.

---

## Key Metrics Glossary

| Metric | Definition |
|---|---|
| **PER** | Player Efficiency Rating — overall per-minute production |
| **WS / WS/48** | Win Shares / Win Shares per 48 minutes |
| **TS%** | True Shooting Percentage (accounts for 2P, 3P, FT) |
| **BPM** | Box Plus/Minus — points contributed per 100 possessions |
| **OBPM / DBPM** | Offensive / Defensive Box Plus/Minus |
| **VORP** | Value Over Replacement Player |
| **USG%** | Usage Percentage — share of team plays used by player |
| **3PAr** | 3-Point Attempt Rate |
| **FTr** | Free Throw Attempt Rate |

---

## Setup

### Prerequisites
- Python 3.8+
- A Kaggle account with an API key configured (`~/.kaggle/kaggle.json`)

### Install Dependencies

```bash
python -m venv .myvenv
source .myvenv/bin/activate       # macOS/Linux
# .myvenv\Scripts\activate        # Windows

pip install pandas kagglehub scikit-learn numpy scipy xgboost nba_api
```

### Run

```bash
python model_training.py
```

This pulls/cleans the data (via `get_and_clean_data.py` and `mvp_full_stats.py`), fits the model, and prints predicted MVP win shares for the 2026 season along with the test-set MSE.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading, filtering, merging |
| `numpy` | Numerical operations |
| `kagglehub` | Kaggle dataset downloads |
| `nba_api` | Live 2025-26 season player stats |
| `scikit-learn` | Preprocessing (`StandardScaler`, `OrdinalEncoder`), pipeline, train/test split, metrics |
| `xgboost` | `XGBRegressor` — the model used to predict MVP voting share |
| `scipy` | Scientific computing |

---

## Status

The project has a working end-to-end pipeline: historical + live data is downloaded and cleaned, an `XGBRegressor` is trained on past MVP winners' stats, and the fitted pipeline produces **predicted MVP win shares for the 2026 season**. Next steps include improving feature selection and validating predictions as the season progresses.
