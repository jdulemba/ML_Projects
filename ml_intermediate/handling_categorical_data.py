import time
tic = time.time()

import pandas as pd
from pdb import set_trace
import utils.Utilities as Utils
from sklearn.model_selection import train_test_split

## read the data and store data in DataFrame titled melbourne_data
data = Utils.csv_to_pandas_DF("melb_data.csv")

# Separate target from predictors
y = data.Price
X = data.drop(["Price"], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Get list of categorical variables
object_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = Utils.fit_ml_model("RandomForest", X_train, y_train, **{"random_state" : 0, "n_estimators" : 100})
    mae = Utils.get_mae(model, X_valid, y_valid)
    
    return mae

"""
Score from Approach 1 (Drop Categorical Variables)
"""
drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

drop_mae = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
print(f"MAE from Approach 1 (Drop categorical variables) = {drop_mae}")

"""
Score from Approach 2 (Ordinal Encoding)
"""
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

ordinal_mae = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
print(f"MAE from Approach 2 (Ordinal Encoding) = {ordinal_mae}")

"""
Score from Approach 3 (One-Hot Encoding)
"""
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

OH_mae = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)
print(f"MAE from Approach 3 (One-Hot Encoding) = {OH_mae}") 


toc = time.time()
print("Total runtime: %.1f" % (toc - tic))
