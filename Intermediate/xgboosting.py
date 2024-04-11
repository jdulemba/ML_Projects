import time
tic = time.time()

"""
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.

It begins by initializing the ensemble with a single model, whose predictions can be pretty naive.
(Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

Then, we start the cycle:
    First, we use the current ensemble to generate predictions for each observation in the dataset.
    To make a prediction, we add the predictions from all models in the ensemble.

    These predictions are used to calculate a loss function (like mean squared error, for instance).
    
    Then, we use the loss function to fit a new model that will be added to the ensemble.
    Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss.
    (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
    
    Finally, we add the new model to ensemble, and ...
    
    ... repeat!


XGBoost has a few parameters that can dramatically affect accuracy and training speed.
The first parameters to understand are:

n_estimators

    n_estimators specifies how many times to go through the modeling cycle described above.
    It is equal to the number of models that we include in the ensemble.
        Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
        
        Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).

    Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

early_stopping_rounds

    early_stopping_rounds offers a way to automatically find the ideal value for n_estimators.
    Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators.
    It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

    Since random chance sometimes causes a single round where validation scores don't improve,
    you need to specify a number for how many rounds of straight deterioration to allow before stopping.
    Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

    When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.

learning_rate

    Instead of getting predictions by simply adding up the predictions from each component model,
    we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.

    This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting.
    If we use early stopping, the appropriate number of trees will be determined automatically.

    In general, a small learning rate and large number of estimators will yield more accurate XGBoost models,
    though it will also take the model longer to train since it does more iterations through the cycle.
    As default, XGBoost sets learning_rate=0.1.

n_jobs

    On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.
    It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't help.

    The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction.
    But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.
"""

import pandas as pd
from pdb import set_trace
import utils.Utilities as Utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

## read the data and store data in DataFrame titled melbourne_data
X = Utils.csv_to_pandas_DF("melb_data.csv")
# Remove rows with missing target, separate target from predictors
X.dropna(axis = 0, subset = ["Price"], inplace = True)
y = X.Price
X.drop(["Price"], axis = 1, inplace = True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join = "left", axis = 1)

#set_trace()
# build and fit model using XGBoost
my_model = XGBRegressor(
    n_estimators = 1000,
    early_stopping_rounds = 5,
    learning_rate = 0.05,
    n_jobs = 4,
    random_state = 0
)
my_model.fit(X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

# make predictions and evaluate the model
predictions = my_model.predict(X_valid)
print(f"Mean Absolute Error: {mean_absolute_error(predictions, y_valid)}")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
