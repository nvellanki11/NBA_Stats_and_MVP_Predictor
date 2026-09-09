from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
 
from mvp_full_stats import mvp_stats_list
from get_and_clean_data import full_stats_pd, mvp_winners, stats2026_pd

# Split data into training and testing sets
from sklearn.model_selection import GroupShuffleSplit

# Preprocessing data
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Pipeline for preprocessing and modeling
from sklearn.pipeline import Pipeline


# Canonical box-score/advanced stats the model is expected to train on. Hard-coded
# (rather than inferred from whatever columns happen to overlap) so that a schema
# change in either data source shows up as a loud failure instead of the model
# silently training on a handful of leftover columns.
FEATURES = [
    'FG%', '3P%', '2P%', 'eFG%', 'FT%', 'TS%',
    'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
    'PER', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP',
    'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'MP',
]

missing_from_full = [f for f in FEATURES if f not in full_stats_pd.columns]
missing_from_2026 = [f for f in FEATURES if f not in stats2026_pd.columns]
missing_features = sorted(set(missing_from_full) | set(missing_from_2026))

# PER, WS, WS/48, OBPM, DBPM, BPM, VORP are Basketball-Reference-only metrics with
# no live-season nba_api equivalent; STL%/BLK%/TOV% aren't exposed by nba_api's
# per-player advanced endpoint either. These 10 are permanently absent from
# stats2026_pd, and 'MP' (vs. full_stats_pd's 'MP_x') is a historical naming quirk
# — none of these are a sign of a schema change, so the threshold accounts for them.
MAX_MISSING_FEATURES = 11
if len(missing_features) > MAX_MISSING_FEATURES:
    raise RuntimeError(
        f"{len(missing_features)}/{len(FEATURES)} canonical features are missing "
        f"from the training and/or prediction data (max allowed: {MAX_MISSING_FEATURES}).\n"
        f"Missing from full_stats_pd: {missing_from_full}\n"
        f"Missing from stats2026_pd: {missing_from_2026}\n"
        "Check get_and_clean_data.py for a column naming/schema change."
    )

overlap_features = [f for f in FEATURES if f not in missing_features]

# Train the model using the selected features and target variable
features = full_stats_pd[overlap_features]  # Assuming these are the target and non-feature columns
target = full_stats_pd['Share']

# Split by season (not by row) so no season's players end up split across both
# train and test - Share is season-relative, so a row-level split would leak
# information about a season's overall MVP race between the two sets.
season_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, test_idx = next(season_splitter.split(features, target, groups=full_stats_pd['Year']))
X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

# Ordinal encode the 'Pos_x' column in the full stats dataset
ordinal_encoder = OrdinalEncoder()
full_stats_pd['Pos_x'] = ordinal_encoder.fit_transform(full_stats_pd[['Pos_x']])

# Create and fit a pipeline that includes preprocessing and modeling steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step to standardize features
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)),  # Select model)
])
pipeline.fit(X_train, y_train)

stats2026_pd = stats2026_pd[stats2026_pd['GP'] >= 60]  # Filter out players with less than 60 games played in the 2025-26 season
X_2026 = stats2026_pd[overlap_features]

# Make predictions on the test set and evaluate the model
predicted_vote_shares = pipeline.predict(X_2026)

# Dataframe that contains player names and predicted vote shares for 2026
results = pd.DataFrame({
    'Player': stats2026_pd['Player'],
    'Predicted Vote Share': predicted_vote_shares
})

# Sort the results by predicted vote share in descending order
results = results.sort_values(by='Predicted Vote Share', ascending=False)

# Display the results
print(results)

# Check error metrics on the test set
y_pred_test = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')

# Backtest: for each of the last ~10 historical seasons, compare the player the
# model ranks #1 by predicted share against the player who actually won MVP.
last_seasons = sorted(full_stats_pd['Year'].unique())[-10:]

backtest_rows = []
for season in last_seasons:
    season_pd = full_stats_pd[full_stats_pd['Year'] == season]
    season_pred_share = pipeline.predict(season_pd[overlap_features])

    predicted_mvp = season_pd.iloc[season_pred_share.argmax()]['Player']
    actual_mvp = season_pd.iloc[season_pd['Share'].values.argmax()]['Player']

    backtest_rows.append({
        'Season': season,
        'Predicted MVP': predicted_mvp,
        'Actual MVP': actual_mvp,
        'Correct': predicted_mvp == actual_mvp,
    })

backtest_results = pd.DataFrame(backtest_rows)

# Display the backtest results
print(backtest_results)