import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset (Replace with actual dataset path or method)
df = pd.read_csv("housing.csv")

# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# Encode categorical features (if any)
df = pd.get_dummies(df, drop_first=True)

# Define features and target (Replace 'actual_target_column_name' with real column name)
X = df.drop(columns=["median_house_value"])  # Assuming 'median_house_value' is the target
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with feature scaling and SVR
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])

# Define hyperparameter grid
param_grid = {
    "svr__kernel": ["linear", "rbf", "poly"],
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.01, 0.1, 1, 5]
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
