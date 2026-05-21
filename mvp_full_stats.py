import pandas as pd
from get_and_clean_data import mvp_winners, full_stats_pd, mvps_pd
import numpy as np

# Check types of columns- print(full_stats.Year.dtypes)

# In case dataset will be updated beyond 2020
#mvp25 = full_stats_pd.loc[(full_stats_pd.Player == "Shai Gilgeous-Alexander") & (full_stats_pd.Year == 2025)]
#mvp24 = full_stats_pd.loc[(full_stats_pd.Player == "Nikola Jokic") & (full_stats_pd.Year == 2024)]
#mvp23 = full_stats_pd.loc[(full_stats_pd.Player == "Joel Embiid") & (full_stats_pd.Year == 2023)]
#mvp22 = full_stats_pd.loc[(full_stats_pd.Player == "Nikola Jokic") & (full_stats_pd.Year == 2022)]
#mvp21 = full_stats_pd.loc[(full_stats_pd.Player == "Nikola Jokic") & (full_stats_pd.Year == 2021)]

# Collect MVPs full stats in a list for efficient concatenation
mvp_stats_list = []
for _, row in mvp_winners.iterrows():
    mvp = full_stats_pd.loc[(full_stats_pd.Player == row.Player) & (full_stats_pd.Year == np.int64(row.year))]
    if mvp.empty:
        print("Data not available at this time")
    mvp_stats_list.append(mvp)

# List of advanced stats of mvp winners to be used as Y variable for model training; index reset to 0..n-1
mvp_stats_list = pd.concat(mvp_stats_list, ignore_index=True)