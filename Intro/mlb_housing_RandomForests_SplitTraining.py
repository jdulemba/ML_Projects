import pandas as pd
from pdb import set_trace
import Utilities as Utils
from sklearn.model_selection import train_test_split

## read the data and store data in DataFrame titled melbourne_data
melbourne_data = Utils.csv_to_pandas_DF("melb_data.csv")

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# select the prediction target
y = melbourne_data.Price

# choose features to include in the machine learning model
melbourne_features = ["Rooms", "Bathroom", "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state to ensure same results each run
# Split data into training and validation data, for both features and target.
# The split is based on a random number generator.
#Supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define and fit model
forest_model = Utils.fit_ml_model("RandomForest", train_X, train_y, **{"random_state" : 1})
# calculate the mean absolute error
mae = Utils.get_mae(forest_model, val_X, val_y)
print(f"Mean Absolute Error = {mae}")
