import time
tic = time.time()

from pdb import set_trace
import utils.Utilities as Utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers, callbacks

## ensure reproducibility based on setting random seed value for several libraries/paths
seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ["PYTHONHASHSEED"]=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)



# Setup plotting
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)

spotify = Utils.csv_to_pandas_DF("spotify.csv")

X = spotify.copy().dropna()
y = X.pop("track_popularity")
artists = X["track_artist"]

features_num = ["danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo", "duration_ms"]
features_cat = ["playlist_genre"]

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# We'll do a "grouped" split to keep all of an artist's songs in one split or the other.
# This is to help prevent signal leakage.
def group_split(X, y, group, train_size = 0.75):
    splitter = GroupShuffleSplit(train_size = train_size, random_state = 0)
    train, test = next(splitter.split(X, y, groups = group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

"""Let's start with the simplest network, a linear model. This model has low capacity."""

model = keras.Sequential([
    layers.Dense(1, input_shape = input_shape),
])
model.compile(
    optimizer = "adam",
    loss = "mae",
)
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 512,
    epochs = 50,
    verbose = 0, # suppress output since we"ll plot the curves
)

print(f"\nMinimum validation loss: {np.min(np.array(history.history['val_loss'])):0.4f}\n")
fig, ax = plt.subplots(dpi = 100)
ax.plot(history.history["loss"], "k", label = "Training Loss")
ax.plot(history.history["val_loss"], "b", label = "Validation Loss")
ax.autoscale()
ax.set_xlim(history.epoch[0], history.epoch[-1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Simple Linear Model")
fname = "results/OverFitting_Underfitting_Exercise_SimpleLinearModel"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
It's not uncommon for the curves to follow a "hockey stick" pattern like you see here.
This makes the final part of training hard to see, so let's start at epoch 10 instead:
"""

fig, ax = plt.subplots(dpi = 100)
ax.plot(history.epoch[10:], history.history["loss"][10:], "k", label = "Training Loss")
ax.plot(history.epoch[10:], history.history["val_loss"][10:], "b", label = "Validation Loss")
ax.autoscale()
ax.set_xlim(history.epoch[10], history.epoch[-1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Simple Linear Model")
fname = "results/OverFitting_Underfitting_Exercise_SimpleLinearModel_After_Epoch10"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
1) Evaluate Baseline

What do you think? Would you say this model is underfitting, overfitting, just right?
    The gap between these curves is quite small and the validation loss never increases, so it's more likely that the network is underfitting than overfitting.
    It would be worth experimenting with more capacity to see if that's the case.

Now let's add some capacity to our network. We'll add three hidden layers with 128 units each.
"""

model = keras.Sequential([
    layers.Dense(128, activation = "relu", input_shape = input_shape),
    layers.Dense(64, activation = "relu"),
    layers.Dense(1)
])
model.compile(
    optimizer = "adam",
    loss = "mae",
)
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 512,
    epochs = 50,
    verbose = 0
)

print(f"\nMinimum validation loss: {np.min(np.array(history.history['val_loss'])):0.4f}\n")

fig, ax = plt.subplots(dpi = 100)
ax.plot(history.history["loss"], "k", label = "Training Loss")
ax.plot(history.history["val_loss"], "b", label = "Validation Loss")
ax.autoscale()
ax.set_xlim(history.epoch[0], history.epoch[-1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Model With 3 Hidden Layers")
fname = "results/OverFitting_Underfitting_Exercise_Model_with_3_Hidden_Layers"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
2) Add Capacity

What is your evaluation of these curves? Underfitting, overfitting, just right?
    Now the validation loss begins to rise very early, while the training loss continues to decrease. This indicates that the network has begun to overfit.
    At this point, we would need to try something to prevent it, either by reducing the number of units or through a method like early stopping.


3) Define Early Stopping Callback

Now define an early stopping callback that waits 5 epochs (patience') for a change in validation loss of at least 0.001 (min_delta) 
and keeps the weights with the best loss (restore_best_weights).
"""

early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 5,
    restore_best_weights = True,
)
model = keras.Sequential([
    layers.Dense(128, activation = "relu", input_shape = input_shape),
    layers.Dense(64, activation = "relu"),
    layers.Dense(1)
])
model.compile(
    optimizer = "adam",
    loss = "mae",
)
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 512,
    epochs = 50,
    verbose = 0,
    callbacks = [early_stopping],
)

print(f"\nMinimum validation loss: {np.min(np.array(history.history['val_loss'])):0.4f}\n")

fig, ax = plt.subplots(dpi = 100)
ax.plot(history.history["loss"], "k", label = "Training Loss")
ax.plot(history.history["val_loss"], "b", label = "Validation Loss")
ax.autoscale()
ax.set_xlim(history.epoch[0], history.epoch[-1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Model With 3 Hidden Layers and Early Stopping")
fname = "results/OverFitting_Underfitting_Exercise_Model_with_3_Hidden_Layers_EarlyStopping"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
4) Train and Interpret

Was this an improvement compared to training without early stopping?
    The early stopping callback did stop the training once the network began overfitting.
    Moreover, by including restore_best_weights we still get to keep the model where validation loss was lowest.
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
