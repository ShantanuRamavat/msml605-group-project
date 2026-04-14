import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Housing.csv")
print("Columns:", list(df.columns))

df = df.drop(columns=['hotwaterheating', 'mainroad'])
df = df.rename(columns={'area': 'area (sq ft)'})

binary_cols = ['guestroom', 'basement', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
df = df[df['price'] < 12000000]

X = df.drop('price', axis=1)
y = df['price']

print("Feature columns:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Run GridSearch to find best params
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40],
    'max_depth': [5, 7, -1]
}

lgbm = lgb.LGBMRegressor(objective='regression', random_state=42)
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best params:", best_params)

# IMPORTANT: create a BRAND NEW model with best params and fit it fresh
# This avoids the GridSearchCV wrapper serialization bug
final_model = lgb.LGBMRegressor(
    objective='regression',
    random_state=42,
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth']
)
final_model.fit(X_train, y_train)

# Evaluate
y_pred = final_model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
acc  = 100 - mape

print(f"R2:       {r2:.4f}")
print(f"MAE:      {mae:.2f}")
print(f"RMSE:     {rmse:.2f}")
print(f"Accuracy: {acc:.2f}%")

# Verify prediction works BEFORE saving
test_row = X_test.iloc[:1]
test_pred = final_model.predict(test_row)
print(f"Test prediction: {test_pred}")

joblib.dump(final_model, "model.pkl")
print("model.pkl saved successfully.")

# Final verification - reload and predict
reloaded = joblib.load("model.pkl")
verify_pred = reloaded.predict(test_row)
print(f"Reloaded model prediction: {verify_pred}")
print("Verification passed!" if abs(test_pred[0] - verify_pred[0]) < 1 else "WARNING: predictions differ!")