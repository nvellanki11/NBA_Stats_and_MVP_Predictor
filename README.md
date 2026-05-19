# NBA MVP Prediction

A data analysis and prediction project focused on identifying NBA Most Valuable Player (MVP) award winners using historical player statistics and MVP voting data.

---

## Overview

This project aggregates decades of NBA player performance metrics alongside MVP voting records to explore patterns in MVP selection and build toward a predictive model. Data spans from the 2001 season (Allen Iverson's MVP win) through the 2025 season (Shai Gilgeous-Alexander).

---

## Project Structure

```
NBA_MVP_Prediction/
├── main.py                        # Data loading, merging, and exploratory analysis
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

### `main.py` — Script Walkthrough

1. **Download datasets** from Kaggle using `kagglehub`.
2. **Load data** — reads all three MVP voting CSVs and concatenates them into a single `mvps_pd` DataFrame; loads the full stats CSV into `full_stats`.
3. **Filter MVP winners** — isolates rows where `Rank == "1"` to get the actual award winners for each season.
4. **Player lookups** — queries `full_stats` for specific MVP winners by name to inspect their season statistics. Currently active for:
   - 2025 MVP: Shai Gilgeous-Alexander
   - 2024 MVP: Nikola Jokic

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

pip install pandas kagglehub scikit-learn numpy scipy
```

### Run

```bash
python main.py
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | 3.0.1 | Data loading, filtering, merging |
| `numpy` | 2.4.3 | Numerical operations |
| `kagglehub` | 1.0.0 | Kaggle dataset downloads |
| `scikit-learn` | 1.8.0 | Machine learning (planned) |
| `scipy` | 1.17.1 | Scientific computing |

---

## Status

The project is currently in the **data exploration phase**. The data pipeline (loading, merging, filtering) is functional. Next steps include feature engineering and training a model to predict MVP award probability from player statistics.
