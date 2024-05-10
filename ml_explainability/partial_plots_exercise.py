import time
tic = time.time()

"""
This script will create partial dependence plots and practice building insights with data from the Taxi Fare Prediction competition.
"""

from pdb import set_trace

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance

# load data and remove data with extreme outlier coordinates or negative fares
data = pd.read_csv("data/nyc_taxi_train.csv", nrows=50000)

data = data.query("pickup_latitude > 40.7 and pickup_latitude < 40.8 and " +
                  "dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and " +
                  "pickup_longitude > -74 and pickup_longitude < -73.9 and " +
                  "dropoff_longitude > -74 and dropoff_longitude < -73.9 and " +
                  "fare_amount > 0"
                  )

y = data.fare_amount

base_features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    #"passenger_count"
]

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=30, random_state=1)
first_model.fit(train_X, train_y)

# plot the partial dependence plot for all features
for feat_name in base_features:
    fig, ax = plt.subplots()
    
    PartialDependenceDisplay.from_estimator(first_model, val_X, [feat_name], ax=ax)
    
    fname = f"results/PartialPlots_Exercise_RFModel_Taxi_{feat_name}_PartialPlot"
    fig.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    plt.close(fig)


"""
Why does the partial dependence plot have this U-shape?

Does your explanation suggest what shape to expect in the partial dependence plots for the other features?
    We have a sense from the permutation importance results that distance is the most important determinant of taxi fare.

    This model didn't include distance measures (like absolute change in latitude or longitude) as features, 
    so coordinate features (like pickup_longitude) capture the effect of distance. 
    Being picked up near the center of the longitude values lowers predicted fares on average, because it means shorter trips (on average).

    For the same reason, we see the general U-shape in all our partial dependence plots.
"""

# 2D partial plot for pickup_longitude and dropoff_longitude
fig, ax = plt.subplots(figsize=(8, 6))

f_names = [("pickup_longitude", "dropoff_longitude")]
PartialDependenceDisplay.from_estimator(first_model, val_X, f_names, ax=ax)

fname = f"results/PartialPlots_Exercise_RFModel_Taxi_{'_'.join(f_names[0])}_PartialPlot"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
You should expect the plot to have contours running along a diagonal. We see that to some extent, though there are interesting caveats.
We expect the diagonal contours because these are pairs of values where the pickup and dropoff longitudes are nearby, indicating shorter trips (controlling for other factors).
As you get further from the central diagonal, we should expect prices to increase as the distances between the pickup and dropoff longitudes also increase.
The surprising feature is that prices increase as you go further to the upper-right of this graph, even staying near that 45-degree line.
This could be worth further investigation, though the effect of moving to the upper right of this graph is small compared to moving away from that 45-degree line.


In the PDP's you've seen so far, location features have primarily served as a proxy to capture distance traveled.
In the permutation importance lessons, the features abs_lon_change and abs_lat_change were added as a more direct measure of distance.

Identify the most important difference between this partial dependence plot and the one you got without absolute value features.
"""

# create new features
data["abs_lon_change"] = abs(data.dropoff_longitude - data.pickup_longitude)
data["abs_lat_change"] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_lat_change",
    "abs_lon_change"
]

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1)
second_model.fit(new_train_X, new_train_y)

feat_name = "pickup_longitude"
fig, ax = plt.subplots()

PartialDependenceDisplay.from_estimator(second_model, val_X, [feat_name], ax=ax)

fname = f"results/PartialPlots_Exercise_SecondRFModel_Taxi_{feat_name}_PartialPlot"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
The difference is that the partial dependence plot became smaller. Both plots have a lowest vertical value of 8.5.
But, the highest vertical value in the top chart is around 10.7, and the highest vertical value in the bottom chart is below 9.1.
In other words, once you control for absolute distance traveled, the pickup_longitude has a smaller impact on predictions.


Consider a scenario where you have only 2 predictive features, which we will call feat_A and feat_B.
Both features have minimum values of -1 and maximum values of 1.
The partial dependence plot for feat_A increases steeply over its whole range,
whereas the partial dependence plot for feature B increases at a slower rate (less steeply) over its whole range.

Does this guarantee that feat_A will have a higher permutation importance than feat_B. Why or why not?
    No. This doesn't guarantee feat_a is more important.
    For example, feat_a could have a big effect in the cases where it varies, but could have a single value 99% of the time.
    In that case, permuting feat_a wouldn't matter much, since most values would be unchanged.



ANOTHER EXAMPLE
The code below does the following:

1) Creates two features, X1 and X2, having random values in the range [-2, 2].
2) Creates a target variable y, which is always 1.
3) Trains a RandomForestRegressor model to predict y given X1 and X2.
4) Creates a PDP plot for X1 and a scatter plot of X1 vs. y.

Do you have a prediction about what the PDP plot will look like?

*** Modify the initialization of y so that the PDP plot has a positive slope in the range [-1,1], and a negative slope everywhere else.
"""

from numpy.random import rand

n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2

# Your code here
# Create y. you should have X1 and X2 in the expression for y
y = np.ones(n_samples)
    ## *** modified code ***
    ## without the next 3 lines the PDP of X1 is flat
y[(X1 >= -1) & (X1 <= 1)] = y[(X1 >= -1) & (X1 <= 1)] + X1[(X1 >= -1) & (X1 <= 1)] # modify y values where X1 is [-1, 1]
y[(X1 < -1)] = y[(X1 < -1)] - X1[(X1 < -1)] # modify y values where X1 < -1
y[(X1 > 1)] = y[(X1 > 1)] - X1[(X1 > 1)] # modify y values where X1 < 1


# create dataframe 
my_df = pd.DataFrame({"X1": X1, "X2": X2, "y": y})
predictors_df = my_df.drop(["y"], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1)
my_model.fit(predictors_df, my_df.y)

# plot PDP
fig, ax = plt.subplots()

PartialDependenceDisplay.from_estimator(my_model, predictors_df, ["X1"], ax=ax)

fname = "results/PartialPlots_Exercise_2FeatureModel_X1_PartialPlot"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
Create a dataset with 2 features and a target, such that the pdp of the first feature is flat, but its permutation importance is high. We will use a RandomForest for the model.
"""

#set_trace()
# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2

# Your code here
# Create y. you should have X1 and X2 in the expression for y
y = X1 * X2


# create dataframe 
my_df = pd.DataFrame({"X1": X1, "X2": X2, "y": y})
predictors_df = my_df.drop(["y"], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1)
my_model.fit(predictors_df, my_df.y)

# plot PDP
fig, ax = plt.subplots()

PartialDependenceDisplay.from_estimator(my_model, predictors_df, ["X1"], ax=ax)

fname = "results/PartialPlots_Exercise_2FeatureModel_X1_PartialPlot_Part2"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


perm = permutation_importance(my_model, predictors_df, my_df.y, random_state=1)
sorted_importances_idx = (-perm.importances_mean).argsort()
importances = pd.DataFrame(
    np.stack((perm.importances_mean[sorted_importances_idx], perm.importances_std[sorted_importances_idx])).T,
    index=predictors_df.columns[sorted_importances_idx],
    columns=["Mean", "Std"],
)

print(f"My model importances:\n{importances}")
"""
importances

        Mean       Std
X2  1.994903  0.019841
X1  1.985946  0.019670
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
