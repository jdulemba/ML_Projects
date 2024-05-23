import time
tic = time.time()

from pdb import set_trace
import pandas as pd
import utils.Utilities as Utils
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight="bold")
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=16, titlepad=10)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)


book_sales = Utils.csv_to_pandas_DF("book_sales.csv", **{"index_col" : "Date", "parse_dates" : ["Date"]}).drop("Paperback", axis = 1)
book_sales["Time"] = np.arange(len(book_sales.index))
book_sales["Lag_1"] = book_sales["Hardcover"].shift(1)
book_sales = book_sales.reindex(columns=["Hardcover", "Time", "Lag_1"])

ar = Utils.csv_to_pandas_DF("ar.csv")

dtype_dict = {
    "store_nbr": "category",
    "family": "category",
    "sales": "float32",
    "onpromotion": "uint64",
}

store_sales = Utils.csv_to_pandas_DF("train.csv", **{"dtype" : dtype_dict, "parse_dates" : ["date"]})
store_sales = store_sales.set_index("date").to_period("D")
store_sales = store_sales.set_index(["store_nbr", "family"], append=True)
average_sales = store_sales.groupby("date").mean()["sales"]

"""
One advantage linear regression has over more complicated algorithms is that the models it creates are explainable,
it's easy to interpret what contribution each feature makes to the predictions.
In the model 
    target = weight * feature + bias
the weight tells you by how much the target changes on average for each unit of change in the feature.
"""

#import statsmodels.api as sm
#def simple_regplot(
#    x, y, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None
#):
#    """ Draw a regression line with error interval. """
#    ax = plt.gca() if ax is None else ax
#
#    # calculate best-fit line and interval
#    x_fit = sm.add_constant(x)
#    fit_results = sm.OLS(y, x_fit).fit()
#
#    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
#    pred = fit_results.get_prediction(eval_x)
#
#    # draw the fit line and error interval
#    ci_kws = {} if ci_kws is None else ci_kws
#    ax.fill_between(
#        eval_x[:, 1],
#        pred.predicted_mean - n_std * pred.se_mean,
#        pred.predicted_mean + n_std * pred.se_mean,
#        alpha=0.5,
#        **ci_kws,
#    )
#    line_kws = {} if line_kws is None else line_kws
#    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)
#
#    # draw the scatterplot
#    scatter_kws = {} if scatter_kws is None else scatter_kws
#    ax.scatter(x, y, c=h[0].get_color(), **scatter_kws)
#
#    return fit_results

## try this method at some point in order to avoid seaborn
model = LinearRegression()
X = book_sales.loc[:, ["Time"]]
y = book_sales.loc[:, "Hardcover"]
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

#set_trace()

fig, ax = plt.subplots()
ax.plot(book_sales["Time"], book_sales["Hardcover"], color="0.75")
ax.plot(book_sales["Time"], y_pred, color="b", alpha=0.5)
ax.scatter(book_sales["Time"], book_sales["Hardcover"], color="0.25")
ax.set_xlabel("Time")
ax.set_ylabel("Hardcover Sales")
ax.set_title("Time Plot of Hardcover Sales")

fname = "results/LinearRegression_Exercise_HardcoverSales"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)



"""
1) Interpret linear regression with the time dummy

The linear regression line has an equation of (approximately) Hardcover = 3.33 * Time + 150.5.
Over 6 days how much on average would you expect hardcover sales to change?
    A change of 6 steps in Time corresponds to an average change of 6 * 3.33 = 19.98 in Hardcover sales.


Interpreting the regression coefficients can help us recognize serial dependence in a time plot.
Consider the model 
    target = weight * lag_1 + error
where error is random noise and weight is a number between -1 and 1.
The weight in this case tells you how likely the next time step will have the same sign as the previous time step:
    a weight close to 1 means target will likely have the same sign as the previous step, while a weight close to -1 means target will likely have the opposite sign.


2) Interpret linear regression with a lag feature

One of these series has the equation target = 0.95 * lag_1 + error and the other has the equation target = -0.95 * lag_1 + error,
differing only by the sign on the lag feature.
Can you tell which equation goes with each series?
The series with the 0.95 weight will tend to have values with signs that stay the same.
The series with the -0.95 weight will tend to have values with signs that change back and forth.
    Series 1 was generated by target = 0.95 * lag_1 + error and Series 2 was generated by target = -0.95 * lag_1 + error.
"""

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar["ar1"])
ax1.set_title("Series 1")
ax2.plot(ar["ar2"])
ax2.set_title("Series 2")

fname = "results/LinearRegression_Exercise_AR"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
3) Fit a time-step feature

Now get started with the Store Sales - Time Series Forecasting competition data.
The entire dataset comprises almost 1800 series recording store sales across a variety of product families from 2013 into 2017. 
For this lesson, just work with a single series (average_sales) of the average sales each day.
"""

df = average_sales.to_frame()
df["Time"] = np.arange(len(df.index))

X = df.loc[:, ["Time"]]
y = df.loc[:, "sales"]

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)


fig, ax = plt.subplots()
ax = y.plot(**plot_params, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title("Time Plot of Total Store Sales")
ax.autoscale()

fname = "results/LinearRegression_Exercise_TotalStoreSales"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
4) Fit a lag feature to Store Sales
"""

df["Lag_1"] = df["sales"].shift(1)
X = df.loc[:, ["Lag_1"]].dropna()  # features
y = df.loc[:, "sales"]  # target
y, X = y.align(X, join="inner")  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax.plot(X["Lag_1"], y, ".", color="0.25")
ax.plot(X["Lag_1"], y_pred)
ax.set(ylabel="sales", xlabel="Lag_1", title="Lag Plot of Average Sales")

fname = "results/LinearRegression_Exercise_LagPlotAverageSales"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))