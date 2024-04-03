import time
tic = time.time()

import pandas as pd
from pdb import set_trace
import utils.Utilities as Utils
from sklearn.model_selection import train_test_split

## read the data and store data in DataFrame titled melbourne_data
data = Utils.csv_to_pandas_DF("melb_data.csv")

# select the prediction target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(["Price"], axis=1)
X = melb_predictors.select_dtypes(exclude=["object"])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


"""
Approach 1:  drop columns with missing values
"""
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

reduced_model = Utils.fit_ml_model("RandomForest", reduced_X_train, y_train, **{"random_state" : 0, "n_estimators" : 10})
reduced_mae = Utils.get_mae(reduced_model, reduced_X_valid, y_valid)
print(f"MAE from Approach 1 (Drop columns with missing values) = {reduced_mae}")


"""
Approach 2:  imputation of missing values using the mean
"""
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

imputed_model = Utils.fit_ml_model("RandomForest", imputed_X_train, y_train, **{"random_state" : 0, "n_estimators" : 10})
imputed_mae = Utils.get_mae(imputed_model, imputed_X_valid, y_valid)
print(f"MAE from Approach 2 (Imputation) = {imputed_mae}")


"""
Approach 3 (Extension to Imputation): impute missing values while keeping track of which values were imputed
"""

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + "_was_missing"] = X_train_plus[col].isnull()
    X_valid_plus[col + "_was_missing"] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

imputed_plus_model = Utils.fit_ml_model("RandomForest", imputed_X_train_plus, y_train, **{"random_state" : 0, "n_estimators" : 10})
imputed_plus_mae = Utils.get_mae(imputed_plus_model, imputed_X_valid_plus, y_valid)
print(f"MAE from Approach 3 (An Extension to Imputation) = {imputed_plus_mae}")


# Shape of training data (num_rows, num_columns)
print(f"\nX_train shape: {X_train.shape}")

# Number of missing values in each column of training data
print("Number of missing values in each columns of training data:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

toc = time.time()
print("Total runtime: %.1f" % (toc - tic))
