# Import necessary libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load your sales data into a pandas dataframe
data = pd.read_csv(r'C:\Users\welcome\Documents\futuresale prediction.csv')

# Define the target variable (sales)
target = "Sales"

# Check if the data is stationary (important for ARIMA)
from statsmodels.tsa.stattools import adfuller
result = adfuller(data[target])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# If not stationary, consider differencing the data
if result[1] > 0.05:
    data[target] = data[target].diff().dropna()
    result = adfuller(data[target])
    print(f"ADF Statistic (after differencing): {result[0]}")
    print(f"p-value (after differencing): {result[1]}")

# Define ARIMA parameters (p, d, q) based on data analysis (e.g., ACF/PACF plots)
model = ARIMA(data[target], order=(1, 1, 1))

# Fit the model
model_fit = model.fit()

# Forecast future sales (adjust the number of periods for your needs)
future_predictions = model_fit.forecast(steps=12)  # Forecast for the next 12 periods

# Evaluate the model (using the first half of predictions for comparison)
actual_sales = data[target][-len(future_predictions):]
predicted_sales = future_predictions[:len(actual_sales)]
mse = mean_squared_error(actual_sales, predicted_sales)
print(f"Mean Squared Error: {mse}")

# Print the predicted sales
print(f"Predicted Sales: {predicted_sales.tolist()}")
