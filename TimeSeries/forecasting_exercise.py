import time
tic = time.time()


from pdb import set_trace

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight="bold")
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from utils.Utilities import make_lags, make_multistep_target

def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette="husl", n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler("color", palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax



store_sales = pd.read_csv(
    "data/train.csv",
    usecols=["store_nbr", "family", "date", "sales", "onpromotion"],
    dtype={
        "store_nbr": "category",
        "family": "category",
        "sales": "float32",
        "onpromotion": "uint32",
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

test = pd.read_csv(
    "data/test.csv",
    dtype={
        "store_nbr": "category",
        "family": "category",
        "onpromotion": "uint32",
    },
    parse_dates=["date"],
)
test["date"] = test.date.dt.to_period("D")
test = test.set_index(["store_nbr", "family", "date"]).sort_index()

"""
Consider the following three forecasting tasks:

a. 3-step forecast using 4 lag features with a 2-step lead time
b. 1-step forecast using 3 lag features with a 1-step lead time
c. 3-step forecast using 4 lag features with a 1-step lead time
"""

N = 10
ts = pd.Series(
    np.arange(N),
    index=pd.period_range(start="2010", freq="Y", periods=N, name="Year"),
    dtype=pd.Int8Dtype,
)

## case a 
y = ts.copy()
case_a_features = make_lags(y, 4, lead_time=2)
y = make_multistep_target(y, steps=3)
case_a_data = pd.concat({"Targets": y, "Features": case_a_features}, axis=1)

"""
      Targets                   Features
     y_step_3 y_step_2 y_step_1  y_lag_2 y_lag_3 y_lag_4 y_lag_5
Year
2010        2        1        0     <NA>    <NA>    <NA>    <NA>
2011        3        2        1     <NA>    <NA>    <NA>    <NA>
2012        4        3        2        0    <NA>    <NA>    <NA>
2013        5        4        3        1       0    <NA>    <NA>
2014        6        5        4        2       1       0    <NA>
2015        7        6        5        3       2       1       0
2016        8        7        6        4       3       2       1
2017        9        8        7        5       4       3       2
2018     <NA>        9        8        6       5       4       3
2019     <NA>     <NA>        9        7       6       5       4
"""

## case b
y = ts.copy()
case_b_features = make_lags(y, 3)
y = make_multistep_target(y, steps=1)
case_b_data = pd.concat({"Targets": y, "Features": case_b_features}, axis=1)

"""
      Targets Features
     y_step_1  y_lag_1 y_lag_2 y_lag_3
Year
2010        0     <NA>    <NA>    <NA>
2011        1        0    <NA>    <NA>
2012        2        1       0    <NA>
2013        3        2       1       0
2014        4        3       2       1
2015        5        4       3       2
2016        6        5       4       3
2017        7        6       5       4
2018        8        7       6       5
2019        9        8       7       6
"""

## case c
y = ts.copy()
case_c_features = make_lags(y, 4)
y = make_multistep_target(y, steps=3)
case_c_data = pd.concat({"Targets": y, "Features": case_c_features}, axis=1)

"""
      Targets                   Features
     y_step_3 y_step_2 y_step_1  y_lag_1 y_lag_2 y_lag_3 y_lag_4
Year
2010        2        1        0     <NA>    <NA>    <NA>    <NA>
2011        3        2        1        0    <NA>    <NA>    <NA>
2012        4        3        2        1       0    <NA>    <NA>
2013        5        4        3        2       1       0    <NA>
2014        6        5        4        3       2       1       0
2015        7        6        5        4       3       2       1
2016        8        7        6        5       4       3       2
2017        9        8        7        6       5       4       3
2018     <NA>        9        8        7       6       5       4
2019     <NA>     <NA>        9        8       7       6       5
"""

"""
Look at the time indexes of the training and test sets. From this information, can you identify the forecasting task for Store Sales?
"""
#print("Training Data", "\n" + "-" * 13 + "\n", store_sales)
#print("\n")
#print("Test Data", "\n" + "-" * 9 + "\n", test)

"""
Training Data
-------------
                                      sales  onpromotion
store_nbr family     date
1         AUTOMOTIVE 2013-01-01   0.000000            0
                     2013-01-02   2.000000            0
                     2013-01-03   3.000000            0
                     2013-01-04   3.000000            0
                     2013-01-05   5.000000            0
...                                    ...          ...
9         SEAFOOD    2017-08-11  23.830999            0
                     2017-08-12  16.859001            4
                     2017-08-13  20.000000            0
                     2017-08-14  17.000000            0
                     2017-08-15  16.000000            0


Test Data
---------
                                       id  onpromotion
store_nbr family     date
1         AUTOMOTIVE 2017-08-16  3000888            0
                     2017-08-17  3002670            0
                     2017-08-18  3004452            0
                     2017-08-19  3006234            0
                     2017-08-20  3008016            0
...                                  ...          ...
9         SEAFOOD    2017-08-27  3022271            0
                     2017-08-28  3024053            0
                     2017-08-29  3025835            0
                     2017-08-30  3027617            0
                     2017-08-31  3029399            0


1) Identify the forecasting task for Store Sales competition

Try to identify the forecast origin and the forecast horizon. How many steps are within the forecast horizon? What is the lead time for the forecast?
    The training set ends on 2017-08-15, which gives us the forecast origin.
    The test set comprises the dates 2017-08-16 to 2017-08-31, and this gives us the forecast horizon.
    There is one step between the origin and horizon, so we have a lead time of one day.

    Put another way, we need a 16-step forecast with a 1-step lead time.
    We can use lags starting with lag 1, and we make the entire 16-step forecast using features from 2017-08-15.


2) Create multistep dataset for Store Sales

Create targets suitable for the Store Sales forecasting task. Use 4 days of lag features. Drop any missing values from both targets and features.
"""

y = family_sales.loc[:, "sales"]

# Make 4 lag features
X = make_lags(y, lags=4, lead_time=1).dropna()

# Make multistep target
y = make_multistep_target(y, steps=16).dropna()

y, X = y.align(X, join="inner", axis=0)

# Now, apply the DirRec strategy to the multiple time series of Store Sales.
le = LabelEncoder()
X = (X
    .stack("family")  # wide to long
    .reset_index("family")  # convert index to column
    .assign(family=lambda x: le.fit_transform(x.family))  # label encode
)
y = y.stack("family")  # wide to long


"""
3) Forecast with the DirRec strategy

Instatiate a model that applies the DirRec strategy to XGBoost.
"""

from sklearn.multioutput import RegressorChain

model = RegressorChain(XGBRegressor())
model.fit(X, y)

y_pred = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
).clip(0.0)


# see a sample of the 16-step predictions this model makes on the training data
FAMILY = "BEAUTY"
START = "2017-04-01"
EVERY = 16

y_pred_ = y_pred.xs(FAMILY, level="family", axis=0).loc[START:]
y_ = family_sales.loc[START:, "sales"].loc[:, FAMILY]

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
y_.plot(**plot_params, ax=ax, alpha=0.5)
plot_multistep(y_pred_, ax=ax, every=EVERY)
ax.legend(["Training Set", "Forecast"])
ax.set(title=f"{FAMILY} Sales", ylabel="Sales")

fname = "results/Forecasting_Exercise_Sales_DirRecStrategy_Forecasting"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
