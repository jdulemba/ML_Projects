import time
tic = time.time()

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight="bold")
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from pathlib import Path
import utils.Utilities as Utils
from pdb import set_trace

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


store_sales = Utils.csv_to_pandas_DF("train.csv",
   **{"dtype" : {
        "store_nbr": "category",
        "family": "category",
        "sales": "float32",
        "onpromotion": "uint32",
       },
       "parse_dates" : ["date"],
       "usecols" : ["store_nbr", "family", "date", "sales", "onpromotion"],
   }
)
store_sales["date"] = store_sales.date.dt.to_period("D")
store_sales = store_sales.set_index(["store_nbr", "family", "date"]).sort_index()

family_sales = (
    store_sales
    .groupby(["family", "date"])
    .mean() 
    .unstack("family")
    .loc["2017", ["sales", "onpromotion"]]
)


"""
Not every product family has sales showing cyclic behavior, and neither does the series of average sales. 
Sales of school and office supplies, however, show patterns of growth and decay not well characterized by trend or seasons. 
In this question and the next, you'll model cycles in sales of school and office supplies using lag features.

Trend and seasonality will both create serial dependence that shows up in correlograms and lag plots.
To isolate any purely cyclic behavior, we'll start by deseasonalizing the series.
Use the code in the next cell to deseasonalize Supply Sales. We'll store the result in a variable y_deseason.
"""

supply_sales = family_sales.loc(axis=1)[:, "SCHOOL AND OFFICE SUPPLIES"]
y = supply_sales.loc[:, "sales"].squeeze()

fourier = CalendarFourier(freq="ME", order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,
    drop=True,
    additional_terms=[fourier],
)
X_time = dp.in_sample()
X_time["NewYearsDay"] = (X_time.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X_time, y)
y_deseason = y - model.predict(X_time)
y_deseason.name = "sales_deseasoned"

## plot deseasonalized sales
fig, ax = plt.subplots()
y_deseason.plot(ax=ax)
ax.set_title("Sales of School and Office Supplies (deseasonalized)")

fname = "results/Features_Exercise_School_and_Office_Supplies_Deseasonalized"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
Does this deseasonalized series show cyclic patterns? 
To confirm our intuition, we can try to isolate cyclic behavior using a moving-average plot just like we did with trend.
The idea is to choose a window long enough to smooth over short-term seasonality, but short enough to still preserve the cycles.


1) Plotting cycles

Create a seven-day moving average from y, the series of supply sales. Use a centered window, but don't set the min_periods argument.
"""

y_ma = y.rolling(
    window = 7,       # 7-day window
    center = True,      # puts the average at the center of the window
    #min_periods = 183,  # choose about half the window size
).mean()

fig, ax = plt.subplots()
y_ma.plot(
    ax=ax, linewidth=3, title="Sales of School and Office Supplies (deseasonalized) - 7-Day Moving Average", legend=False
)

fname = "results/Features_Exercise_School_and_Office_Supplies_Deseasonalized_7Day_MovingAverage"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")

"""
Do you see how the moving average plot resembles the plot of the deseasonalized series? In both, we can see cyclic behavior indicated.
Let's examine our deseasonalized series for serial dependence. Take a look at the partial autocorrelation correlogram and lag plot.
"""

## make partial correlation plots
fig = plot_pacf(y_deseason, lags=8)
fname = "results/Features_Exercise_School_and_Office_Supplies_Deseasonalized_PACFPlots"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

## make lag plots
fig = plot_lags(y_deseason, lags=8, nrows=2)

fname = "results/Features_Exercise_School_and_Office_Supplies_Deseasonalized_LagPlots"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
2) Examine serial dependence in Store Sales

Are any of the lags significant according to the correlogram?
Does the lag plot suggest any relationships that weren't apparent from the correlogram?
    The correlogram indicates the first lag is likely to be significant, as well as possibly the eighth lag.
    The lag plot suggests the effect is mostly linear.
"""


"""
Recall from the tutorial that a leading indicator is a series whose values at one time can be used to predict the target at a future time -- 
a leading indicator provides "advance notice" of changes in the target.

The competition dataset includes a time series that could potentially be useful as a leading indicator -- the onpromotion series, 
which contains the number of items on a special promotion that day.
Since the company itself decides when to do a promotion, there's no worry about "lookahead leakage";
we could use Tuesday's onpromotion value to forecast sales on Monday, for instance.

Examine leading and lagging values for onpromotion plotted against sales of school and office supplies.
"""

onpromotion = supply_sales.loc[:, "onpromotion"].squeeze().rename("onpromotion")

# Drop days without promotions
fig = plot_lags(x=onpromotion.loc[onpromotion > 1], y=y_deseason.loc[onpromotion > 1], lags=3, nrows=1)
#fig = plot_lags(x=onpromotion.loc[onpromotion > 1], y=y_deseason.loc[onpromotion > 1], lags=3, leads=3, nrows=1)

fname = "results/Features_Exercise_Sales_Deseasonalized_vs_Onpromotion_LagPlots"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
3) Examine time series features

Does it appear that either leading or lagging values of onpromotion could be useful as a feature?
    The lag plot indicates that both leading and lagged values of onpromotion are correlated with supply sales.
    This suggests that both kinds of values could be useful as features. There may be some non-linear effects as well.


4) Create time series features

Create the features indicated in the solution to Question 3.
"""

X_lags = make_lags(y_deseason, lags=1)

X_promo = pd.concat([
    make_lags(onpromotion, lags=1),
    onpromotion,
    #make_leads(onpromotion, leads=1),
], axis=1)

X = pd.concat([X_time, X_lags, X_promo], axis=1).dropna()
y, X = y.align(X, join="inner")


## use predictions from the resulting model
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=30, shuffle=False, random_state=0)

model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=X_train.index).clip(0.0)
y_pred = pd.Series(model.predict(X_valid), index=X_valid.index).clip(0.0)

rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
rmsle_valid = mean_squared_log_error(y_valid, y_pred) ** 0.5
print(f"Training RMSLE: {rmsle_train:.5f}")
print(f"Validation RMSLE: {rmsle_valid:.5f}")

fig, ax = plt.subplots()
y.plot(ax=ax, **plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
y_fit.plot(ax=ax, label="Fitted", color="C0")
y_pred.plot(ax=ax, label="Forecast", color="C3")
ax.legend()

fname = "results/Features_Exercise_AverageSales_Fit_Forecast"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
Winners of Kaggle forecasting competitions have often included moving averages and other rolling statistics in their feature sets.
Such features seem to be especially useful when used with GBDT algorithms like XGBoost.

In Lesson 2 you learned how to compute moving averages to estimate trends. 
Computing rolling statistics to be used as features is similar except we need to take care to avoid lookahead leakage. 
First, the result should be set at the right end of the window instead of the center --
that is, we should use center=False (the default) in the rolling method. Second, the target should be lagged a step.


5) Create statistical features

Create the following features
    14-day rolling median (median) of lagged target
    7-day rolling standard deviation (std) of lagged target
    7-day sum (sum) of items "on promotion", with centered window
"""

y_lag = supply_sales.loc[:, "sales"].shift(1)
onpromo = supply_sales.loc[:, "onpromotion"]

# 28-day mean of lagged target
mean_7 = y_lag.rolling(7).mean()
# YOUR CODE HERE: 14-day median of lagged target
median_14 = y_lag.rolling(14).median()
# YOUR CODE HERE: 7-day rolling standard deviation of lagged target
std_7 = y_lag.rolling(7).std()
# YOUR CODE HERE: 7-day sum of promotions with centered window
promo_7 = onpromo.rolling(7, center=True).sum()

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
