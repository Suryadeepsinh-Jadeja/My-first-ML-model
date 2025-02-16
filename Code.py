import pandas as pd

# Load the dataset from a CSV file
# Assumes 'Sweden4.csv' contains a dataset with a 'Price' column as the target variable
df = pd.read_csv('file_address_in_.csv_format')

# Separate the target variable (Price) from the dataset
y = df['Price']

# Remove 'Price' from feature set and encode categorical variables with one-hot encoding
x = df.drop(['Price'], axis=1)
x = pd.get_dummies(x)  # Convert categorical variables into binary (dummy) variables

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Import Linear Regression model from scikit-learn
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model and train it on the training data
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict target values for both training and testing sets
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Import evaluation metrics for regression models
from sklearn.metrics import mean_squared_error, r2_score

# Compute Mean Squared Error (MSE) and R-squared (R2) for training data
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# Compute MSE and R2 for testing data
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Store Linear Regression results in a DataFrame
lr_results = pd.DataFrame(['LinearRegression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create and train a Random Forest model with a max depth of 2
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Predict target values for both training and testing sets using Random Forest
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# Compute MSE and R2 for training data using Random Forest
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

# Compute MSE and R2 for testing data using Random Forest
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# Store Random Forest results in a DataFrame
rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# Combine results from both models into a single DataFrame
df_models = pd.concat([lr_results, rf_results], axis=0)

# Reset index to clean up DataFrame
df_models.reset_index(drop=True)

# Import necessary libraries for visualization
import matplotlib.pyplot as plt
import numpy as np

# Plot actual vs predicted prices for training data using Linear Regression
plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)  # Scatter plot of actual vs predicted values

# Fit a trend line (linear fit) to the scatter plot
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), '#F7866D')  # Plot the trend line

# Label the axes
plt.ylabel('Predicted Price')
plt.xlabel('Experimental Price')

# Plot actual vs predicted prices for test data using Linear Regression
plt.figure(figsize=(5,5))
plt.scatter(x=y_test, y=y_lr_test_pred, c="#7CAE00", alpha=0.3)  # Scatter plot of actual vs predicted values

# Fit a trend line to the scatter plot
z = np.polyfit(y_test, y_lr_test_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), '#F7866D')  # Plot the trend line

# Label the axes
plt.ylabel('Predicted Price')
plt.xlabel('Experimental Price')
