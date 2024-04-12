import time
tic = time.time()

"""
Construct the full pipeline in three steps.
"""
import numpy as np
#import pandas as pd
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



## testing different parameter values by changing the n_estimators value and then plot
def get_score(n_estimators):
    """Return the average MAE over 5 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(
        steps = [
            ("preprocessor", SimpleImputer()),
            ("model", RandomForestRegressor(n_estimators = n_estimators, random_state = 0))
        ]
    )
    scores = -1 * cross_val_score(my_pipeline, X, y, cv = 5, scoring ="neg_mean_absolute_error")
    return scores.mean()


#set_trace()
    # specify n_estimators values
nEst_min, nEst_max, nEst_step = 800., 1500., 100.
#nEst_min, nEst_max, nEst_step = 100., 1000., 100.
nEst_array = np.arange(nEst_min, nEst_max + nEst_step, nEst_step, dtype=int)
scores_array = np.array([[i, get_score(i)] for i in nEst_array])

# find n_estimators value corresponding to minimum of MAE mean
nEst_mae_min = scores_array[np.argmin(scores_array[:, 1]), 0]
print(f"MAE min = {np.min(scores_array[:, 1]): .4f} occurs when using {nEst_mae_min} estimators")

## plot results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.subplots_adjust(hspace=.07)
ax.plot(list(scores_array[:, 0]), list(scores_array[:, 1]), **{"color" : "k"})
ax.axvline(nEst_mae_min, color="k", linestyle="--")
ax.autoscale()
#ax.set_xlim(0., np.max(nEst_array))
ax.set_ylabel("Mean Absolute Error Average")
ax.set_xlabel("Number of Trees")
ax.set_title("Results using Random Forest")
fname = "results/RandomForest_nEstimators_Optimization"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


toc = time.time()
print("Total runtime: %.1f" % (toc - tic))
