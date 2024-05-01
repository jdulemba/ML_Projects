import time
tic = time.time()

from pdb import set_trace
import pandas as pd
import utils.Utilities as Utils
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess

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

retail_sales = Utils.csv_to_pandas_DF("us-retail-sales.csv", **{"parse_dates" : ["Month"], "index_col" : "Month"}).to_period("D")
food_sales = retail_sales.loc[:, "FoodAndBeverage"]
auto_sales = retail_sales.loc[:, "Automobiles"]

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
1) Determine trend with a moving average plot

The US Retail Sales dataset contains monthly sales data for a number of retail industries in the United States.
"""

sales_trend = food_sales.rolling(
    window = 12,       # 12-month window
    center = True,      # puts the average at the center of the window
    min_periods = 6,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

fig, ax = plt.subplots()
food_sales.plot(**plot_params, alpha=0.5, ax=ax)
sales_trend.plot(ax=ax, linewidth=3)
ax.set(xlabel="Date", ylabel="Sales", title="Food Sales - 12-Month Moving Average")

fname = "results/Trend_Exercise_FoodSales_MovingAvg"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
2) Identify trend

What order polynomial trend might be appropriate for the Food and Beverage Sales series?
Can you think of a non-polynomial curve that might work even better?
    The upwards bend in the trend suggests an order 2 (quadratic) polynomial might be appropriate.

    If you've worked with economic time series before, you might guess that the growth rate in Food and Beverage Sales is best expressed as a percent change. 
    Percent change can often be modeled using an exponential curve.
"""

# see a moving average plot of average_sales estimating the trend.
avg_sales_trend = average_sales.rolling(
    window = 365,
    center = True,
    min_periods = 183,
).mean()

fig, ax = plt.subplots()
average_sales.plot(**plot_params, alpha=0.5, ax=ax)
avg_sales_trend.plot(ax=ax, linewidth=3)
ax.set(xlabel="Date", ylabel="Sales", title="Average Food Sales - Moving Average")

fname = "results/Trend_Exercise_AvgFoodSales_MovingAvg"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
3) Create a Trend Feature

Use DeterministicProcess to create a feature set for a cubic trend model. Also create features for a 90-day forecast.
"""

y = average_sales.copy()  # the target

dp = DeterministicProcess(
    index=average_sales.index,  # dates from the training data
    constant=False,       # dummy feature for the bias (y_intercept)
    order=3,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()
X_fore = dp.out_of_sample(steps=90)

## plot trend and its forecast
model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

fig, ax = plt.subplots()
y.plot(**plot_params, alpha=0.5, ax=ax)
y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.set(xlabel="Date", ylabel="Average Item Sales", title=f"Average Food Sales - Order {dp._order} Trend Forecast")
ax.legend()

fname = f"results/Trend_Exercise_AvgFoodSales_Order{dp._order}TrendForecast"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
One way to fit more complicated trends is to increase the order of the polynomial you use.
To get a better fit to the somewhat complicated trend in Store Sales, we could try using an order 11 polynomial.
"""

dp = DeterministicProcess(index=y.index, order=11)
X = dp.in_sample()
X_fore = dp.out_of_sample(steps=90)

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

fig, ax = plt.subplots()
y.plot(**plot_params, alpha=0.5, ax=ax)
y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.set(xlabel="Date", ylabel="Average Item Sales", title=f"Average Food Sales - Order {dp._order} Trend Forecast")
ax.legend()

fname = f"results/Trend_Exercise_AvgFoodSales_Order{dp._order}TrendForecast"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
4) Understand risks of forecasting with high-order polynomials

High-order polynomials are generally not well-suited to forecasting, however. Can you guess why?
    An order 11 polynomial will include terms like t ** 11.
    Terms like these tend to diverge rapidly outside of the training period making forecasts very unreliable.
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
