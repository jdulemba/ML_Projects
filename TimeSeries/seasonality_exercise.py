import time
tic = time.time()

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 4), titlesize=18, titleweight="bold")
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)

import seaborn as sns

from pdb import set_trace
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
#from warnings import simplefilter
#simplefilter("ignore")
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import utils.Utilities as Utils

# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    #set_trace()
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# Load holiday dataset
holidays_events = Utils.csv_to_pandas_DF("holidays_events.csv",
    **{"dtype" : {
        "type": "category",
        "locale": "category",
        "locale_name": "category",
        "description": "category",
        "transferred": "bool",
        },
       "parse_dates" : ["date"],
       #"infer_datetime_format" : True,
    }
)
holidays_events = holidays_events.set_index("date").to_period("D")

store_sales = Utils.csv_to_pandas_DF("train.csv",
   **{"dtype" : {
        "store_nbr": "category",
        "family": "category",
        "sales": "float32",
       },
       "parse_dates" : ["date"],
       "usecols" : ["store_nbr", "family", "date", "sales"],
       #"infer_datetime_format" : True,
   }
)
store_sales["date"] = store_sales.date.dt.to_period("D")
store_sales = store_sales.set_index(["store_nbr", "family", "date"]).sort_index()

average_sales = (
    store_sales
    .groupby("date").mean()
    .squeeze()
    .loc["2017"]
)

## examine the seasonal plot
X = average_sales.to_frame()
X["week"] = X.index.week
X["day"] = X.index.dayofweek
fig, ax = plt.subplots(figsize=(11, 6))
seasonal_plot(X, y="sales", period="week", freq="day", ax=ax)

fname = "results/Seasonality_Exercise_AverageSales_SeasonalPlot"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

## plot periodogram
fig, ax = plt.subplots(figsize=(11, 6))
plot_periodogram(average_sales, ax=ax)

fname = "results/Seasonality_Exercise_AverageSales_PeriodogramFrequencies"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
1) Determine Seasonality

What kind of seasonality do you see evidence of?
    Both the seasonal plot and the periodogram suggest a strong weekly seasonality. 
    From the periodogram, it appears there may be some monthly and biweekly components as well.
    In fact, the notes to the Store Sales dataset say wages in the public sector are paid out biweekly, on the 15th and last day of the month -- a possible origin for these seasons.
"""

"""
2) Create seasonal features

Use DeterministicProcess and CalendarFourier to create:

    1) indicators for weekly seasons and
    2) Fourier features of order 4 for monthly seasons.
"""

y = average_sales.copy()

fourier = CalendarFourier(freq="ME", order=4)  # 4 sin/cos pairs for "M"onthly seasonality

dp = DeterministicProcess(
    index=y.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index

    ## fit model
model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold", ax=ax)
y_pred.plot(ax=ax, label="Seasonal")
ax.legend()

fname = "results/Seasonality_Exercise_AverageSales_SeasonalTrend"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
Removing from a series its trend or seasons is called detrending or deseasonalizing the series.

Look at the periodogram of the deseasonalized series.
"""

#set_trace()
y_deseason = y - y_pred

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
plot_periodogram(y, ax=ax1)
ax1.set_title("Product Sales Frequency Components")
plot_periodogram(y_deseason, ax=ax2)
ax2.set_title("Deseasonalized")

fname = "results/Seasonality_Exercise_AverageSales_Deseasonalized"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
3) Check for remaining seasonality

Based on these periodograms, how effectively does it appear your model captured the seasonality in Average Sales?
Does the periodogram agree with the time plot of the deseasonalized series?
    The periodogram for the deseasonalized series lacks any large values.
    By comparing it to the periodogram for the original series, we can see that our model was able to capture the seasonal variation in Average Sales.
"""

"""
The Store Sales dataset includes a table of Ecuadorian holidays.
"""

# National and regional holidays in the training set
holidays = (
    holidays_events
    .query("locale in ['National', 'Regional']")
    .loc["2017":"2017-08-15", ["description"]]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)

"""From a plot of the deseasonalized Average Sales, it appears these holidays could have some predictive power."""

fig, ax = plt.subplots()
y_deseason.plot(**plot_params, ax=ax)
plt.plot_date(holidays.index, y_deseason[holidays.index], color='C3')
ax.set_title('National and Regional Holidays')

fname = "results/Seasonality_Exercise_AverageSales_Deseasonalized_with_Holidays"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
4) Create holiday features

What kind of features could you create to help your model make use of this information?
"""

X_holidays = pd.get_dummies(holidays)
X2 = X.join(X_holidays, on='date').fillna(0.0)

"""Fit the seasonal model with holiday features added. Do the fitted values seem to have improved?"""

model = LinearRegression()
model.fit(X2, y)

y_pred = pd.Series(model.predict(X2), index=X2.index)

fig, ax = plt.subplots()
y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold", ax=ax)
y_pred.plot(ax=ax, label="Seasonal")
ax.legend()

fname = "results/Seasonality_Exercise_AverageSales_SeasonalTrend_with_Holidays"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
