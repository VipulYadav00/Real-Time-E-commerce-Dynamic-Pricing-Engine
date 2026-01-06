import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data you just created
df = pd.read_csv('processed_data.csv')

top_product = df.groupby('product_id')['total_orders'].sum().idxmax()
product_data = df[df['product_id']== top_product]

plt.figure(figsize=(10,6))
sns.scatterplot(data=product_data, x='price', y='total_orders')
plt.title(f'Price vs Demand for Product: {top_product}')
plt.xlabel('Price ($)')
plt.ylabel('Number of Orders')
plt.show()

