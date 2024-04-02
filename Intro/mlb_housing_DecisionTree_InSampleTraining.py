import pandas as pd
from pdb import set_trace

## read the data and store data in DataFrame titled melbourne_data
from Utilities import csv_to_pandas_DF
melbourne_data = csv_to_pandas_DF("melb_data.csv")
# print a summary of the data in Melbourne data
melbourne_data.describe()

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# select the prediction target
y = melbourne_data.Price

# choose features to include in the machine learning model
melbourne_features = ["Rooms", "Bathroom", "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
X.describe()

# An example of defining a decision tree model with scikit-learn and fitting it with the features and target variable.
from sklearn.tree import DecisionTreeRegressor


# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# make predictions on first few rows of training data
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

### validate our "In-Sample" model with certain metrics
print(f"Predicted error = {y.head().values - melbourne_model.predict(X.head())}")

# calculate the mean absolute error 
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print(f"Mean Absolute Error = {mean_absolute_error(y, predicted_home_prices)}")
#set_trace()
