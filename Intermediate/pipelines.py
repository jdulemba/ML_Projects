import time
tic = time.time()

"""
Construct the full pipeline in three steps.
"""

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

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

"""
Step 1: Define preprocessing steps

Similar to how a pipeline bundles together preprocessing and modeling steps,
we use the ColumnTransformer class to bundle together different preprocessing steps.
The code below:
    imputes missing values in numerical data, and
    imputes missing values and applies a one-hot encoding to categorical data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy = "constant")

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers = [
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


"""
Step 2: Define the model

Next, we define a random forest model with the familiar RandomForestRegressor class.
"""

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 100, random_state = 0)


"""
Step 3: Create and Evaluate the Pipeline

Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps.
There are a few important things to notice:
    With the pipeline, we preprocess the training data and fit the model in a single line of code.
    (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps.
    This becomes especially messy if we have to deal with both numerical and categorical variables!)

    With the pipeline, we supply the unprocessed features in X_valid to the predict() command,
    and the pipeline automatically preprocesses the features before generating predictions.
    (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)
"""

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(
    steps = [
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print(f"MAE: {score}")


toc = time.time()
print("Total runtime: %.1f" % (toc - tic))
