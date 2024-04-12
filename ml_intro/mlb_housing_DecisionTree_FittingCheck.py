import time
tic = time.time()

import numpy as np
from pdb import set_trace
import utils.Utilities as Utils
from sklearn.model_selection import train_test_split

## read the data and store data in DataFrame titled melbourne_data
melbourne_data = Utils.csv_to_pandas_DF("melb_data.csv")
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# select the prediction target
y = melbourne_data.Price

# choose features to include in the machine learning model
melbourne_features = ["Rooms", "Bathroom", "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
X.describe()


"""
Define model. Specify a number for random_state to ensure same results each run
Split data into training and validation data, for both features and target.
The split is based on a random number generator.
Supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script.

In order to balance the model parameters between over- and underfiitting,
we want to find the tree depth which corresponds to the minimum of the MAE for the validation data.
"""
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


    # specify leaf_node values
nleaf_min, nleaf_max, nleaf_step = 5., 4000., 5.
max_leaf_node_array = np.arange(nleaf_min, nleaf_max + nleaf_step, nleaf_step, dtype=int)
    # init 2D array to save [leaf_node value, mae value]
nleaf_mae_array = np.zeros((max_leaf_node_array.size, 2))
# compare MAE with differing values of max_leaf_nodes and find minimum
for idx, max_leaf_nodes in enumerate(max_leaf_node_array):
    model = Utils.fit_ml_model("DecisionTree", train_X, train_y, **{"random_state" : 0, "max_leaf_nodes" : max_leaf_nodes})
    model_mae = Utils.get_mae(model, val_X, val_y)
    nleaf_mae_array[idx] = [max_leaf_nodes, model_mae]

# find leaf_node value corresponding to minimum of MAE
nleaf_mae_min = nleaf_mae_array[np.argmin(nleaf_mae_array[:, 1]), 0]
print(f"MAE min = {np.min(nleaf_mae_array[:, 1])} occurs when using {nleaf_mae_min} leafs")

## plot MAE as a function of nleafs
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.subplots_adjust(hspace=.07)
ax.step(nleaf_mae_array[:, 0], nleaf_mae_array[:, 1], where="post", **{"label" : f"Validation Data\nMAE Min = {int(np.min(nleaf_mae_array[:, 1]))} at {int(nleaf_mae_min)}", "color" : "k"})
ax.axvline(nleaf_mae_min, color="k", linestyle="--")
ax.autoscale()
ax.set_xlim(0., nleaf_max + nleaf_step)
ax.set_ylabel("Mean Absolute Error")
ax.set_xlabel("Tree Depth")
ax.legend(loc="upper right")
ax.set_title("Results using Decision Trees")
fname = "results/DecisionTree_UnderFitting_OverFitting_Check"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

toc = time.time()
print("Total time: %.1f" % (toc - tic))



