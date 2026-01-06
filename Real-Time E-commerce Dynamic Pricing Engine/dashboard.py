import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="AI Dynamic Pricing Engine", layout="wide")
st.title("📈 Real-Time E-commerce Dynamic Pricing Engine")
st.markdown("This AI-powered tool predicts demand and optimizes price to maximize revenue.")

# 2. Load Model
model = joblib.load('pricing_model.pkl')

# 3. Sidebar for Inputs
st.sidebar.header("Product & Market Conditions")
base_price = st.sidebar.slider("Current Product Price ($)", 10.0, 200.0, 70.0)
prev_sales = st.sidebar.number_input("Yesterday's Sales (Units)", 0, 50, 5)
rolling_sales = st.sidebar.number_input("7-Day Avg Sales", 0.0, 50.0, 4.5)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=0)
is_weekend = st.sidebar.checkbox("Is it a Weekend?")

# 4. Optimization Logic
potential_prices = np.linspace(base_price * 0.5, base_price * 1.5, 20)
results = []

for p in potential_prices:
    # Build feature array for the model
    # Features: [price, day_of_week, month, is_weekend, prev_day_sales, rolling_3d_sales, price_vs_avg]
    input_data = [[p, 0, month, int(is_weekend), prev_sales, rolling_sales, p/base_price]]
    pred_demand = model.predict(input_data)[0]
    revenue = p * pred_demand
    results.append([p, pred_demand, revenue])

res_df = pd.DataFrame(results, columns=['Price', 'Demand', 'Revenue'])
best_row = res_df.loc[res_df['Revenue'].idxmax()]

# 5. Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Recommended Price", f"${best_row['Price']:.2f}")
col2.metric("Predicted Daily Sales", f"{best_row['Demand']:.2f} units")
col3.metric("Expected Daily Revenue", f"${best_row['Revenue']:.2f}")

# 6. Visualization
st.subheader("Revenue Optimization Curve")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res_df['Price'], res_df['Revenue'], color='green', linewidth=2, label='Revenue')
ax.axvline(best_row['Price'], color='red', linestyle='--', label='Optimal Price')
ax.set_xlabel("Price ($)")
ax.set_ylabel("Expected Revenue ($)")
ax.legend()
st.pyplot(fig)

st.success("The model suggests an optimal price based on learned price elasticity!")