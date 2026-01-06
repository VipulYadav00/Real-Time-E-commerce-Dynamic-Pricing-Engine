import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load the features
df = pd.read_csv('final_features.csv')

# 2. Select Features (X) and Target (y)
# We drop product_id and date because the model needs numbers, not strings
features = ['price', 'day_of_week', 'month', 'is_weekend', 
            'prev_day_sales', 'rolling_3d_sales', 'price_vs_avg']
X = df[features]
y = df['total_orders']

# 3. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train XGBoost
# n_estimators=100 means it builds 100 trees to make a decision
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 5. Evaluate the Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Training Complete!")
print(f"Mean Absolute Error: {mae:.2f} (Average error in order prediction)")
print(f"R2 Score: {r2:.2f} (Higher is better, 1.0 is perfect)")

# 6. Save the model for our API later
joblib.dump(model, 'pricing_model.pkl')
print("Model saved as pricing_model.pkl")