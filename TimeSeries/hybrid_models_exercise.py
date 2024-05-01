import time
tic = time.time()

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight="bold")
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)

from pdb import set_trace
#import utils.Utilities as Utils

from pathlib import Path
from warnings import simplefilter
simplefilter("ignore")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor

store_sales = pd.read_csv(
    "data/train.csv",
    usecols=["store_nbr", "family", "date", "sales", "onpromotion"],
    dtype={
        "store_nbr": "category",
        "family": "category",
        "sales": "float32",
    },
    parse_dates=["date"],
)
store_sales["date"] = store_sales.date.dt.to_period("D")
store_sales = store_sales.set_index(["store_nbr", "family", "date"]).sort_index()

family_sales = (
    store_sales
    .groupby(["family", "date"])
    .mean()
    .unstack("family")
    .loc["2017"]
)

"""
In the next two questions, you'll create a boosted hybrid for the Store Sales dataset by implementing a new Python class.
Run this cell to create the initial class definition. You'll add fit and predict methods to give it a scikit-learn like interface.
"""

# You'll add fit and predict methods to this minimal class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method

"""
1) Define fit method for boosted hybrid

Complete the fit definition for the BoostedHybrid class.
"""

def fit(self, X_1, X_2, y):
    # fit self.model_1
    self.model_1.fit(X_1, y)

    y_fit = pd.DataFrame(
        # make predictions with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
    )

    # compute residuals
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze() # wide to long

    # fit self.model_2 on residuals
    self.model_2.fit(X_2, y_resid)

    # Save column names for predict method
    self.y_columns = y.columns
    # Save data for question checking
    self.y_fit = y_fit
    self.y_resid = y_resid


# Add method to class
BoostedHybrid.fit = fit

"""
2) Define predict method for boosted hybrid

Now define the predict method for the BoostedHybrid class.
"""

def predict(self, X_1, X_2):
    y_pred = pd.DataFrame(
        # predict with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long

    # add self.model_2 predictions to y_pred
    y_pred += self.model_2.predict(X_2)

    return y_pred.unstack()  # long to wide


# Add method to class
BoostedHybrid.predict = predict

"""Use the BoostedHybrid class to create a model for the Store Sales data."""

# Target series
y = family_sales.loc[:, "sales"]

# X_1: Features for Linear Regression
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()

# X_2: Features for XGBoost
X_2 = family_sales.drop("sales", axis=1).stack()  # onpromotion feature

# Label encoding for "family"
le = LabelEncoder()  # from sklearn.preprocessing
X_2 = X_2.reset_index("family")
X_2["family"] = le.fit_transform(X_2["family"])

# Label encoding for seasonality
X_2["day"] = X_2.index.day  # values are day of the month


"""
3) Train boosted hybrid

Create the hybrid model by initializing a BoostedHybrid class with LinearRegression() and XGBRegressor() instances.
"""

# Create LinearRegression + XGBRegressor hybrid with BoostedHybrid
model = BoostedHybrid(model_1=LinearRegression(), model_2=XGBRegressor())

# Fit and predict
model.fit(X_1, X_2, y)
y_pred = model.predict(X_1, X_2)

y_pred = y_pred.clip(0.0)


"""
Depending on the problem, you might want to use other hybrid combinations than the linear regression + XGBoost hybrid created.
The following are other algorithms from scikit-learn.
"""

# Model 1 (trend)
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Boosted Hybrid
# Different combinations of the algorithms above
model = BoostedHybrid(
    model_1=Ridge(),
    model_2=KNeighborsRegressor(),
)

"""See the predictions the hybrid model makes."""

y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train, X1_valid = X_1[: "2017-07-01"], X_1["2017-07-02" :]
X2_train, X2_valid = X_2.loc[:"2017-07-01"], X_2.loc["2017-07-02":]

# Some of the algorithms above do best with certain kinds of preprocessing on the features (like standardization)
model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

families = y.columns[0:6]
axs = y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(11, 9), **plot_params, alpha=0.5,
)
y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color="C0", ax=axs)
y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color="C3", ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)

fname = "results/HybridModels_Exercise_StoreSales"
plt.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close()


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
