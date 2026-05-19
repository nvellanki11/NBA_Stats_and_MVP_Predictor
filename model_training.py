from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
from mvp_full_stats import mvp_stats_list
from get_and_clean_data import full_stats_pd, mvp_winners, stats2026_pd

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Preprocessing data
from sklearn.preprocessing import StandardScaler

# Pipeline for preprocessing and modeling
from sklearn.pipeline import Pipeline


# Create and fit a pipeline that includes preprocessing and modeling steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step to standardize features
    ('model', RandomForestRegressor()),  # Modeling step
])

# Retrain and fit the pipeline with training data
features = full_stats_pd.drop(columns=['Player', 'Year', 'Share'])  # Assuming these are the target and non-feature columns
target = full_stats_pd['Share']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
pipeline.fit(X_train, y_train)