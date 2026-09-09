from xgboost import XGBRegressor
from scipy.stats import spearmanr
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

# MSE is a poor headline metric here: most player-seasons have Share = 0, so a
# model that predicts near-zero everywhere scores well on MSE while being
# useless for the thing we actually care about - ranking the MVP race. Instead,
# evaluate on held-out seasons using two rank-aware metrics: whether the actual
# MVP lands in the model's predicted top-3, and Spearman correlation restricted
# to players who received any actual MVP votes (Share > 0).
TOP_K = 3

test_seasons_pd = full_stats_pd.iloc[test_idx]
y_pred_test = pipeline.predict(X_test)
test_results_pd = test_seasons_pd.assign(**{'Predicted Vote Share': y_pred_test})

backtest_rows = []
for season, season_pd in test_results_pd.groupby('Year'):
    actual_mvp = season_pd.loc[season_pd['Share'].idxmax(), 'Player']
    top_k_players = season_pd.nlargest(TOP_K, 'Predicted Vote Share')['Player'].tolist()

    vote_getters = season_pd[season_pd['Share'] > 0]
    if len(vote_getters) >= 2:
        season_spearman = spearmanr(vote_getters['Share'], vote_getters['Predicted Vote Share']).statistic
    else:
        season_spearman = float('nan')

    backtest_rows.append({
        'Season': season,
        'Actual MVP': actual_mvp,
        f'Top-{TOP_K} Predicted': top_k_players,
        'In Top-K': actual_mvp in top_k_players,
        'Spearman (vote-getters)': season_spearman,
    })

backtest_results = pd.DataFrame(backtest_rows)

# Display the backtest results
print(backtest_results)

n_seasons = len(backtest_results)
n_top_k_correct = backtest_results['In Top-K'].sum()
mean_spearman = backtest_results['Spearman (vote-getters)'].mean()
print(
    f"\nCorrectly ranked the actual MVP in the predicted top-{TOP_K} in "
    f"{n_top_k_correct} of {n_seasons} held-out seasons."
)
print(f"Mean Spearman correlation among vote-getters: {mean_spearman:.3f}")