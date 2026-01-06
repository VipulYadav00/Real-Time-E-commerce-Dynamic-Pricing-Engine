📈 Real-Time E-commerce Dynamic Pricing Engine

An end-to-end Machine Learning solution designed to optimize product pricing by predicting consumer demand and maximizing daily revenue.

🚀 Project Overview
This project solves the "Optimal Pricing" problem for e-commerce retailers. Using historical sales data, the system predicts how many units will sell at various price points and recommends the price that yields the highest total revenue.

🛠️ Tech Stack
Languages: Python
ML Framework: XGBoost (Gradient Boosted Decision Trees)
Data Engineering:** Pandas, NumPy, Scikit-learn
API: FastAPI (Backend)
Dashboard: Streamlit (Frontend)
Visualization: Matplotlib, Seaborn

📊 Model Performance
Algorithm: XGBoost Regressor
Mean Absolute Error (MAE): 0.21
R2 Score: 0.45
Features Used: Price, Day of Week, Month, Weekend Indicator, Previous Day Sales, Rolling 3-Day Average Sales.

🏗️ System Architecture
1.Data Prep: Merged raw Olist E-commerce data into a time-series format.

2.Feature Engineering: Developed lag features and rolling averages to capture sales momentum.

3.Training: Optimized an XGBoost model to handle non-linear price elasticity.

4.Optimization: Implemented a revenue maximization algorithm that iterates through price simulations.

5.Interface: Built a Streamlit UI for real-time "What-If" analysis.

🏃 How to Run
1.Install requirements: `pip install -r requirements.txt`

2.Run the Dashboard: `streamlit run dashboard.py`

3.Access the API: `uvicorn app:app --reload`