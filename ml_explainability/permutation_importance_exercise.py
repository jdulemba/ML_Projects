import time
tic = time.time()

"""
This script calculates the permutation importance with a sample of data from the Taxi Fare Prediction competition.
"""

from pdb import set_trace

# Loading data, dividing, modeling and EDA below
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

base_features = ["pickup_longitude",
                 "pickup_latitude",
                 "dropoff_longitude",
                 "dropoff_latitude",
                 "passenger_count"]

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1)
first_model.fit(train_X, train_y)

"""
train_X.describe()

       pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude  passenger_count
count      23466.000000     23466.000000       23466.000000      23466.000000     23466.000000
mean         -73.976827        40.756931         -73.975359         40.757434         1.662320
std            0.014625         0.018206           0.015930          0.018659         1.290729
min          -73.999999        40.700013         -73.999999         40.700020         0.000000
25%          -73.987964        40.744901         -73.987143         40.745756         1.000000
50%          -73.979629        40.758076         -73.978588         40.758542         1.000000
75%          -73.967797        40.769602         -73.966459         40.770406         2.000000
max          -73.900062        40.799952         -73.900062         40.799999         6.000000


train_y.describe()

count    23466.000000
mean         8.472539
std          4.609747
min          0.010000
25%          5.500000
50%          7.500000
75%         10.100000
max        165.000000



The first model uses the following features

    1) pickup_longitude
    2) pickup_latitude
    3) dropoff_longitude
    4) dropoff_latitude
    5) passenger_count

Which variables seem potentially useful for predicting taxi fares? Do you think permutation importance will necessarily identify these features as important?
     It would be helpful to know whether New York City taxis vary prices based on how many passengers they have.
     Most places do not change fares based on numbers of passengers. If you assume New York City is the same, then only the top 4 features listed should matter. 
     At first glance, it seems all of those should matter equally.
"""

first_perm = permutation_importance(first_model, val_X, val_y, random_state=1)
sorted_importances_idx = (-first_perm.importances_mean).argsort()
importances = pd.DataFrame(
    np.stack((first_perm.importances_mean[sorted_importances_idx], first_perm.importances_std[sorted_importances_idx])).T,
    index=val_X.columns[sorted_importances_idx],
    columns=["Mean", "Std"],
)

print(f"First model importances:\n{importances}")
"""
importances

                       Mean       Std
dropoff_latitude   0.864552  0.015499
pickup_latitude    0.847749  0.021203
pickup_longitude   0.622955  0.019903
dropoff_longitude  0.538663  0.026984
passenger_count   -0.001152  0.001588



Before seeing these results, we might have expected each of the 4 directional features to be equally important.
But, on average, the latitude features matter more than the longititude features. Can you come up with any hypotheses for this?
    1) Travel might tend to have greater latitude distances than longitude distances. 
    If the longitudes values were generally closer together, shuffling them wouldn't matter as much. 
    2) Different parts of the city might have different pricing rules (e.g. price per mile), and pricing rules could vary more by latitude than longitude. 
    3) Tolls might be greater on roads going North<->South (changing latitude) than on roads going East <-> West (changing longitude). 
    Thus latitude would have a larger effect on the prediction because it captures the amount of the tolls.


Without detailed knowledge of New York City, it's difficult to rule out most hypotheses about why latitude features matter more than longitude.
A good next step is to disentangle the effect of being in certain parts of the city from the effect of total distance traveled.
The code below creates new features for longitudinal and latitudinal distance. It then builds a model that adds these new features to those you already created.
"""

# create new features
data["abs_lon_change"] = abs(data.dropoff_longitude - data.pickup_longitude)
data["abs_lat_change"] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ["pickup_longitude",
               "pickup_latitude",
               "dropoff_longitude",
               "dropoff_latitude",
               "abs_lat_change",
               "abs_lon_change"]

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1)
second_model.fit(new_train_X, new_train_y)

second_perm = permutation_importance(second_model, new_val_X, new_val_y, random_state=1)
sorted_importances_idx = (-second_perm.importances_mean).argsort()
importances = pd.DataFrame(
    np.stack((second_perm.importances_mean[sorted_importances_idx], second_perm.importances_std[sorted_importances_idx])).T,
    index=new_val_X.columns[sorted_importances_idx],
    columns=["Mean", "Std"],
)

print(f"Second model importances:\n{importances}")
"""
importances

                       Mean       Std
abs_lat_change     0.585122  0.016736
abs_lon_change     0.460798  0.023033
pickup_latitude    0.088731  0.009352
pickup_longitude   0.075311  0.014183
dropoff_latitude   0.068480  0.005236
dropoff_longitude  0.065695  0.004963


How would you interpret these importance scores? Distance traveled seems far more important than any location effects.
But the location still affects model predictions, and pickup location now matters slightly more than dropoff location.
    

A colleague observes that the values for abs_lon_change and abs_lat_change are pretty small (all values are between -0.1 and 0.1),
whereas other variables have larger values.
Do you think this could explain why those coordinates had larger permutation importance values in this case?

Consider an alternative where you created and used a feature that was 100X as large for these features,
and used that larger feature for training and importance calculations. Would this change the outputted permutaiton importance values?
    The scale of features does not affect permutation importance per se.
    The only reason that rescaling a feature would affect PI is indirectly,
    if rescaling helped or hurt the ability of the particular learning method we're using to make use of that feature. 
    That won't happen with tree based models, like the Random Forest used here, but might be able to be affected with Ridge Regression.
    That said, the absolute change features have high importance because they capture total distance traveled, which is the primary determinant of taxi fares...
    It is not an artifact of the feature magnitude.


You've seen that the feature importance for latitudinal distance is greater than the importance of longitudinal distance. 
From this, can we conclude whether travelling a fixed latitudinal distance tends to be more expensive than traveling the same longitudinal distance?
    We cannot tell from the permutation importance results whether traveling a fixed latitudinal distance is more or less expensive than traveling the same longitudinal distance.
    Possible reasons latitude feature are more important than longitude features:
        1) latitudinal distances in the dataset tend to be larger 
        2) it is more expensive to travel a fixed latitudinal distance 
        3) Both of the above. If abs_lon_change values were very small, longitudes could be less important to the model even if the cost per mile of travel in that direction were high.
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
