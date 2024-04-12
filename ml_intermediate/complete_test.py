import time
tic = time.time()

import pandas as pd
from pdb import set_trace
import utils.Utilities as Utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
## do a little pre-processing for missing data using one hot encoder and imputation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV

## read the data and store data in DataFrame titled melbourne_data
X = Utils.csv_to_pandas_DF("melb_data.csv")
# Remove rows with missing target
X.dropna(axis = 0, subset = ["Price"], inplace = True)

## plot percentage of missing values for each column in dataframe
fig, ax = plt.subplots()
fig.subplots_adjust(hspace=.07)
X.isnull().mean().plot.bar(figsize=(20,4), fontsize = 10)
ax.autoscale()
ax.set_ylabel("Number of Missing Values (%)")
ax.set_xlabel("Categories")
fname = "results/MissingValuesPercentage"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

# Delete columns containing either 40% or more than 40% NaN Values
perc = 40.0
min_count =  int(((100-perc)/100)*X.shape[0] + 1)

#X.isnull().mean()[X.isnull().mean() >= 0.4]
#  BuildingArea    0.474963
X = X.dropna(axis=1, thresh=min_count)


# creation of correlation matrix (for numeric data only)
corrM = X.corr(numeric_only=True)

## visualization of the correlation matrix
fig = plt.figure(figsize=(24, 15))
plt.matshow(corrM, fignum=fig.number)
plt.xticks(range(corrM.select_dtypes(["number"]).shape[1]), corrM.select_dtypes(["number"]).columns, fontsize=12, rotation=45)
plt.yticks(range(corrM.select_dtypes(["number"]).shape[1]), corrM.select_dtypes(["number"]).columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.title("Correlation Matrix", fontsize=16);
fname = "results/CorrelationMatrix"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

# which features are highly correlated with price?
# feature selection based on correlation with the output variable
cor_target = abs(corrM["Price"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.1]
correlated_cols = relevant_features.keys().to_list()
# correlated_cols = ["Rooms", "Price", "Distance", "Postcode", "Bedroom2", "Bathroom", "Car", "YearBuilt", "Lattitude", "Longtitude"]
# only Landsize and Propertycount aren't highly correlated with Price


# Separate target from predictors
y = X.Price
#X = X[correlated_cols]
X.drop(["Price"], axis = 1, inplace = True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# X_train_full.select_dtypes(["object"]).nunique()
#   Suburb           308
#   Address        10742
#   Type               3
#   Method             5
#   SellerG          251
#   Date              58
#   CouncilArea       33
#   Regionname         8
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Select numeric columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# Preprocessing for numerical data - a simple imputation strategy
numerical_transformer = SimpleImputer(strategy="median")

# Preprocessing for categorical data
#    imputes missing values and applies a one-hot encoding to categorical data.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# combine the preprocessing of numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Various hyper-parameters to tune for XGBoost model, using grid search CV on the pipeline
model_xgb = XGBRegressor(random_state = 0)
my_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model_xgb)
    ]
)

# parameter grids
parameters =  {
    ##"model__max_depth" : [5, 10],
    ##"model__subsample" : [0.4, 0.7],
    ##"model__colsample_bytree" : [0.7],
    ##"model__reg_lambda" :  [0.8],
    "model__n_estimators" : [1000, 3000],
    #"model__early_stopping_rounds" : [5, 10],
    "model__learning_rate" : [0.05, 0.01],
}

xgb_grid = GridSearchCV(my_pipeline, parameters, cv = 5, n_jobs = 5, verbose=True)
xgb_grid.fit(X_train, y_train)
print(f"Best score = {xgb_grid.best_score_}")
print(f"Best parameters: {xgb_grid.best_params_}")

# using the best parameters from grid search CV 
print("\nUsing optimized parameters for new model\n")

# make predictions and evaluate the model
grid_predictions = xgb_grid.predict(X_valid)
print("\nResults directly from grid")
print(f"Mean Absolute Error: {mean_absolute_error(grid_predictions, y_valid)}")
print(f"Mean Squared Log Error: {mean_squared_log_error(grid_predictions, y_valid)}")
#print("\n\n")
#
#
####
#
##double checking best fit results by hardcoding them
#
####
#optimized_model = XGBRegressor(
#    n_estimators = 3000,
#    random_state = 0,
#    learning_rate = 0.01,
#    eval_metric = "rmsle",
#)
#optimized_pipeline = Pipeline(
#    steps=[
#        ("preprocessor", preprocessor),
#        ("model", optimized_model)
#    ]
#)
#optimized_pipeline.fit(X_train, y_train)
#
## make predictions and evaluate the model
#optimized_predictions = optimized_pipeline.predict(X_valid)
#print("\nResults hardcoding best parameters")
#print(f"Mean Absolute Error: {mean_absolute_error(optimized_predictions, y_valid)}")
#print(f"Mean Squared Log Error: {mean_squared_log_error(optimized_predictions, y_valid)}")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
