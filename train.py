import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import xgboost as xgb

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/house_data.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Drop unnecessary columns
df = df.drop(['street', 'country'], axis=1)

# Feature engineering
df['house_age'] = 2025 - df['yr_built']
df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
df['total_sqft'] = df['sqft_living'] + df['sqft_basement']

# Define features & target
X = df.drop(['price', 'date', 'yr_built', 'yr_renovated'], axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Define preprocessing
# -------------------------------
categorical = ['city', 'statezip']
numerical = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical),
        ('num', 'passthrough', numerical)
    ]
)

# -------------------------------
# Train models
# -------------------------------

# 1. Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# 2. Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# 3. XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

print("Linear Regression:", evaluate(y_test, y_pred_lr))
print("Random Forest:", evaluate(y_test, y_pred_rf))
print("XGBoost:", evaluate(y_test, y_pred_xgb))

# -------------------------------
# Save best model (Random Forest)
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf_pipeline, "models/house_price_model.pkl")

print("✅ Model pipeline saved at models/house_price_model.pkl")
