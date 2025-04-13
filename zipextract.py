import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# Load corrected dataset
file_path = "C:/Users/akshi/Downloads/corrected_dataset.csv"
df = pd.read_csv(file_path)

# Debug: Check column names
print("Columns in dataset:", df.columns)

# Select relevant features
features = ['year', 'chapter', 'subject']
target = 'marks'

# Encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[features]).toarray()

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features))

# Concatenate with original dataset
df_final = pd.concat([encoded_df, df[target]], axis=1)

# Split data
X = df_final.drop(columns=[target])
y = df_final[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'colsample_bytree': [0.8, 1.0],
    'subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

# Best model from tuning
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Calculate mean absolute error on test data
y_pred_test = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Mean Absolute Error on test data: {mae}")

# Create a dataset for 2025 predictions
future_data = df[['chapter', 'subject']].drop_duplicates().copy()
future_data['year'] = 2025

# Encode future data
future_encoded_features = encoder.transform(future_data[features]).toarray()
future_encoded_df = pd.DataFrame(future_encoded_features, columns=encoder.get_feature_names_out(features))

# Predict for 2025
y_future_pred = best_model.predict(future_encoded_df)

# Ensure predictions are multiples of 4
y_future_pred = np.round(y_future_pred / 4) * 4

# Save 2025 predictions
future_data['predicted_marks'] = y_future_pred
future_predictions_path = 'C:/Users/akshi/Downloads/jee_2025_predictions.csv'
future_data.to_csv(future_predictions_path, index=False)

print(f"Predictions for 2025 JEE saved successfully at {future_predictions_path}")
