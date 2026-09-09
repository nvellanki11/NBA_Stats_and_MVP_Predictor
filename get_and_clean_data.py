import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
import unicodedata
from nba_api.stats.endpoints import leaguedashplayerstats

# Function to be applied to dataset to standardize spellings
def remove_accents(text):
    replacements = {'ø': 'o', 'ł': 'l', 'ß': 'ss'}
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore') .decode('utf-8')


# Download latest version
all_stats_path = kagglehub.dataset_download("orelsorekml/nba-player-stats-including-mvp-share")
mvps_path = kagglehub.dataset_download("parthdande/nba-mvp-voting-dataset-2000-2021")

stats_filepath = "/Users/nishantvellanki/Documents/NBA_MVP_Prediction/original_data/NBA_stats_including_MVP.csv"
mvp1_filepath = "/Users/nishantvellanki/Documents/NBA_MVP_Prediction/original_data/2001-2010 MVP Data.csv"
mvp2_filepath = "/Users/nishantvellanki/Documents/NBA_MVP_Prediction/original_data/2010-2021 MVP Data.csv"
mvp3_filepath = "/Users/nishantvellanki/Documents/NBA_MVP_Prediction/original_data/2022-2023 MVP Data.csv"

# Cleaning and reuploading full player stats dataset
full_stats_pd = pd.read_csv(stats_filepath, index_col=False)
full_stats_pd = full_stats_pd.drop_duplicates(keep='last')
full_stats_pd['Player'] = full_stats_pd['Player'].apply(remove_accents)
if not os.path.exists('cleaned_data/cleaned_full_stats.csv'): # save to a new file only once
    full_stats_pd.to_csv('cleaned_data/cleaned_full_stats.csv', index=False)


# Cleaning and reuploading MVP dataset 1 (2001-2010)
mvp1_pd = pd.read_csv(mvp1_filepath, index_col=False)
mvp1_pd = mvp1_pd.drop_duplicates(keep='last')
mvp1_pd['Player'] = mvp1_pd['Player'].apply(remove_accents)
if not os.path.exists('cleaned_data/cleaned_first_mvp_set.csv'): # save to a new file only once
    mvp1_pd.to_csv('cleaned_data/cleaned_first_mvp_set.csv', index=False)


# Cleaning and reuploading MVP dataset 2 (2010-2021)
mvp2_pd = pd.read_csv(mvp2_filepath, index_col=False)
mvp2_pd = mvp2_pd.drop_duplicates(keep='last')
mvp2_pd['Player'] = mvp2_pd['Player'].apply(remove_accents)
if not os.path.exists('cleaned_data/cleaned_second_mvp_set.csv'):
    mvp2_pd.to_csv('cleaned_data/cleaned_second_mvp_set.csv', index=False)


# Cleaning and reuploading MVP dataset 3 (2022-2023)
mvp3_pd = pd.read_csv(mvp3_filepath, index_col=False) #Not updated so not concated
mvp3_pd = mvp3_pd.drop_duplicates(keep='last')
mvp3_pd['Player'] = mvp3_pd['Player'].apply(remove_accents)
if not os.path.exists('cleaned_data/cleaned_third_mvp_set.csv'):
    mvp3_pd.to_csv('cleaned_data/cleaned_third_mvp_set.csv', index=False)

# Concat and create a new 0..n-1 index; drop cross-file duplicates (e.g. 2010 appears in both mvp1 and mvp2)
mvps_pd = pd.concat([mvp1_pd, mvp2_pd], ignore_index=True).drop_duplicates(keep='last')



# Get 2025-26 stats from nba_api - Advanced measure type has the shooting-efficiency
# and usage metrics, Base has the box-score counting stats. Neither has PER/WS/BPM/VORP;
# those are Basketball-Reference-only metrics with no live-season nba_api equivalent.
stats2026_advanced_pd = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame', measure_type_detailed_defense='Advanced', season_type_all_star='Regular Season')
stats2026_advanced_pd = stats2026_advanced_pd.get_data_frames()[0]

stats2026_base_pd = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame', measure_type_detailed_defense='Base', season_type_all_star='Regular Season')
stats2026_base_pd = stats2026_base_pd.get_data_frames()[0]

base_cols = ['PLAYER_ID', 'FG3M', 'FG3A', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
stats2026_pd = stats2026_advanced_pd.merge(stats2026_base_pd[base_cols], on='PLAYER_ID', how='left')

# Clean 2025-26 stats dataset
stats2026_pd = stats2026_pd.drop_duplicates(keep='last')
stats2026_pd = stats2026_pd.rename(columns={'PLAYER_NAME': 'Player', 'TEAM_ABBREVIATION': 'Tm'})
stats2026_pd['Player'] = stats2026_pd['Player'].apply(remove_accents)
stats2026_pd = stats2026_pd.drop(columns=['PLAYER_ID', 'TEAM_ID', 'NICKNAME'])

# Derive 2-point FG% from the base counting stats (nba_api doesn't expose it directly)
two_pt_makes = stats2026_pd['FGM'] - stats2026_pd['FG3M']
two_pt_attempts = stats2026_pd['FGA'] - stats2026_pd['FG3A']
stats2026_pd['2P%'] = (two_pt_makes / two_pt_attempts).where(two_pt_attempts > 0)

# Adjust column names to match those in full_stats_pd
col_mapping = {
    'EFG_PCT':  'eFG%',
    'TS_PCT':   'TS%',
    'USG_PCT':  'USG%',
    'OREB_PCT': 'ORB%',
    'DREB_PCT': 'DRB%',
    'REB_PCT':  'TRB%',
    'AST_PCT':  'AST%',
    'FGA':      'FGA',
    'FG_PCT':   'FG%',
    'FG3_PCT':  '3P%',
    'FT_PCT':   'FT%',
    'REB':      'TRB',
    'MIN':      'MP',
}
stats2026_pd = stats2026_pd.rename(columns=col_mapping)

# Always refresh: unlike the historical datasets, this is live in-season data
# that changes every time the pipeline runs.
stats2026_pd.to_csv('cleaned_data/cleaned_2026_stats.csv', index=False)


# All winners of mvp award from 2001-2023 (AI-Embiid)
mvp_winners = mvps_pd.loc[mvps_pd.Rank == "1"].reset_index(drop=True)