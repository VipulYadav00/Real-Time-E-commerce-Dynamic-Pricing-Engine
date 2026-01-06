import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model
model = joblib.load('pricing_model.pkl')

def suggest_best_price(base_features):
    """
    base_features: a dictionary containing day_of_week, month, etc.
    We will loop through different prices to find the one that maximizes revenue.
    """
    potential_prices = np.linspace(50, 150, 21) # Testing prices from $50 to $150
    results = []

    for price in potential_prices:
        # Prepare the input for the model
        # Order: [price, day_of_week, month, is_weekend, prev_day_sales, rolling_3d_sales, price_vs_avg]
        input_data = pd.DataFrame([[
            price, 
            base_features['day_of_week'],
            base_features['month'],
            base_features['is_weekend'],
            base_features['prev_day_sales'],
            base_features['rolling_3d_sales'],
            price / 100 # Rough estimate for price_vs_avg
        ]], columns=['price', 'day_of_week', 'month', 'is_weekend', 'prev_day_sales', 'rolling_3d_sales', 'price_vs_avg'])
        
        # Predict demand for this price
        predicted_demand = model.predict(input_data)[0]
        
        # Calculate Revenue = Price * Demand
        revenue = price * predicted_demand
        results.append((price, predicted_demand, revenue))

    # Convert to DataFrame to find the max
    pricing_df = pd.DataFrame(results, columns=['Price', 'Predicted_Demand', 'Expected_Revenue'])
    best_option = pricing_df.loc[pricing_df['Expected_Revenue'].idxmax()]
    
    return best_option

# 2. Test it with some dummy data
test_features = {
    'day_of_week': 5, # Saturday
    'month': 12,      # December
    'is_weekend': 1,
    'prev_day_sales': 2,
    'rolling_3d_sales': 1.5
}

recommendation = suggest_best_price(test_features)
print("--- Pricing Recommendation ---")
print(f"Recommended Price: ${recommendation['Price']:.2f}")
print(f"Predicted Demand: {recommendation['Predicted_Demand']:.2f} units")
print(f"Expected Revenue: ${recommendation['Expected_Revenue']:.2f}")