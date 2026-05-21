from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
 
from mvp_full_stats import mvp_stats_list
from get_and_clean_data import full_stats_pd, mvp_winners, stats2026_pd

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Preprocessing data
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Pipeline for preprocessing and modeling
from sklearn.pipeline import Pipeline


# Select features that are present in both datasets for training and prediction
training_features = [c for c in full_stats_pd.columns
    if c not in ('Player', 'Year', 'Share', 'Unnamed: 0', 'Pts Won', 'Pts Max', 'Tm')]
overlap_features = [c for c in training_features if c in stats2026_pd.columns]

# Train the model using the selected features and target variable
features = full_stats_pd[overlap_features]  # Assuming these are the target and non-feature columns
target = full_stats_pd['Share']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

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
predicted_win_shares = pipeline.predict(X_2026)

# Dataframe that contains player names and predicted win shares for 2026
results = pd.DataFrame({
    'Player': stats2026_pd['Player'],
    'Predicted Win Share': predicted_win_shares
})

# Sort the results by predicted win share in descending order
results = results.sort_values(by='Predicted Win Share', ascending=False)

# Display the results
print(results)

# Check error metrics on the test set
y_pred_test = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')