import pandas as pd

#Load Dataset
product = pd.read_csv('data/olist_products_dataset.csv')
item = pd.read_csv('data/olist_order_items_dataset.csv')
orders = pd.read_csv('data/olist_orders_dataset.csv')

# 2. Merge them together
# We want to see: What product was bought, at what price, and when?
df = pd.merge(item, orders, on='order_id')
df = pd.merge(df, product, on='product_id')

# 3. Convert dates to datetime objects
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# 4. Extract Date features (Simple Time-Series prep)
df['order_date'] = df['order_purchase_timestamp'].dt.date

# 5. Group by product and date to see "Daily Sales"
# This is our target: How many items of Product X sold on Day Y at Price Z?
daily_sales = df.groupby(['product_id', 'order_date', 'price']).size().reset_index(name='total_orders')

print(daily_sales.head())
print(f"Dataset shape: {daily_sales.shape}")

daily_sales.to_csv('processed_data.csv', index=False)

print("File Save Successful as Processed_data.csv")