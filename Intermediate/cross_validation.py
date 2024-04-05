import time
tic = time.time()

"""
Construct the full pipeline in three steps.
"""

import pandas as pd
from pdb import set_trace
import utils.Utilities as Utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

## read the data and store data in DataFrame titled melbourne_data
data = Utils.csv_to_pandas_DF("melb_data.csv")

# Select subset of predictors
cols_to_use = ["Rooms", "Distance", "Landsize", "BuildingArea", "YearBuilt"]
X = data[cols_to_use]

# Separate target from predictors
y = data.Price

# Bundle preprocessing and modeling code in a pipeline
"""
Step 1: Define preprocessing steps

Similar to how a pipeline bundles together preprocessing and modeling steps,
we use the ColumnTransformer class to bundle together different preprocessing steps.
The code below:
    imputes missing values in numerical data, and
    imputes missing values and applies a one-hot encoding to categorical data.

Step 2: Define the model

Next, we define a random forest model with the familiar RandomForestRegressor class.

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
model = RandomForestRegressor(n_estimators = 50, random_state = 0)
my_pipeline = Pipeline(
    steps = [
        ("preprocessor", SimpleImputer()),
        ("model", model)
    ]
)


"""
We obtain the cross-validation scores with the cross_val_score() function from scikit-learn.
We set the number of folds with the cv parameter.
"""
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y, cv = 5, scoring ="neg_mean_absolute_error")
print("MAE scores:\n", scores)

print(f"Average MAE score (across experiments): {scores.mean()}")

toc = time.time()
print("Total runtime: %.1f" % (toc - tic))
