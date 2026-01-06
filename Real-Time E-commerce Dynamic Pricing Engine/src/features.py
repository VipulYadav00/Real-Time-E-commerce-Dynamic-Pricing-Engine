import pandas as pd
import numpy as np

# 1. Load the processed data
try:
    df = pd.read_csv('processed_data.csv')
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: processed_data.csv not found. Run data_prep.py first.")
    exit()

# 2. Convert to datetime (Essential for time-based sorting)
df['order_date'] = pd.to_datetime(df['order_date'])

# --- THE MISSING CRITICAL STEP: SORTING ---
# If you don't sort, your 'prev_day_sales' will be from a random product!
df = df.sort_values(by=['product_id', 'order_date'])

# 3. Time-Based Features
df['day_of_week'] = df['order_date'].dt.dayofweek
df['month'] = df['order_date'].dt.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 4. Lag Features
# shift(1) takes the 'total_orders' from the previous row
df['prev_day_sales'] = df.groupby('product_id')['total_orders'].shift(1).fillna(0)

# 5. Rolling Averages
# window=3 means it looks at the last 3 days of sales for that specific product
df['rolling_3d_sales'] = df.groupby('product_id')['total_orders'].transform(
    lambda x: x.rolling(window=3).mean()
).fillna(0)

# 6. Price vs Product Average
# This tells the model if the current price is a 'deal' or a 'hike'
product_avg_price = df.groupby('product_id')['price'].transform('mean')
df['price_vs_avg'] = df['price'] / product_avg_price

# 7. Final Cleanup
# Drop the product_id for the ML model (it's too unique/high cardinality)
# But keep it for now so we can save it.
df.to_csv('final_features.csv', index=False)

print("--- Feature Engineering Complete ---")
print(df[['product_id', 'order_date', 'price', 'total_orders', 'prev_day_sales', 'rolling_3d_sales']].head(10))

