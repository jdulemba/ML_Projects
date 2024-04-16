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

"""
1) Add Dropout to Spotify Model

Here is the last model from Exercise 4.
Add two dropout layers, one after the Dense layer with 128 units, and one after the Dense layer with 64 units. Set the dropout rate on both to 0.3.
"""

model = keras.Sequential([
    layers.Input(input_shape),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation = "relu"),
    layers.Dropout(0.3),
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
ax.set_title("Spotify Model with Dropout=0.3")
fname = "results/Dropout_Batch_Normalization_Exercise_SpotifyModel_Dropout_0p3_Only"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
2) Evaluate Dropout

Recall from Exercise 4 that this model tended to overfit the data around epoch 5. Did adding dropout seem to help prevent overfitting this time?
    From the learning curves, you can see that the validation loss remains near a constant minimum even though the training loss continues to decrease.
    So we can see that adding dropout did prevent overfitting this time.
    Moreover, by making it harder for the network to fit spurious patterns, dropout may have encouraged the network to seek out more of the true patterns,
    possibly improving the validation loss some as well.
"""

"""
Load the Concrete dataset. We won't do any standardization this time. This will make the effect of batch normalization much more apparent.
"""

concrete = Utils.csv_to_pandas_DF("concrete.csv")

df = concrete.copy()

df_train = df.sample(frac = 0.7, random_state = 0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop("CompressiveStrength", axis = 1)
X_valid = df_valid.drop("CompressiveStrength", axis = 1)
y_train = df_train["CompressiveStrength"]
y_valid = df_valid["CompressiveStrength"]

input_shape = [X_train.shape[1]]

# train the default network on the Concrete data
model = keras.Sequential([
    layers.Input(input_shape),
    layers.Dense(512, activation = "relu"),
    layers.Dense(512, activation = "relu"),
    layers.Dense(512, activation = "relu"),
    layers.Dense(1),
])
model.compile(
    optimizer = "sgd", # SGD is more sensitive to differences of scale
    loss = "mae",
    metrics = ["mae"],
)
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 64,
    epochs = 100,
    verbose = 0,
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
ax.set_title("Default Concrete Model")
fname = "results/Dropout_Batch_Normalization_Exercise_ConcreteModel_Default"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
Did you end up with a blank graph? Trying to train this network on this dataset will usually fail.
Even when it does converge (due to a lucky weight initialization), it tends to converge to a very large number.


3) Add Batch Normalization Layers

Batch normalization can help correct problems like this.

Add four BatchNormalization layers, one before each of the dense layers. (Remember to move the input_shape argument to the new first layer.)
"""


# train the batch normalized network on the Concrete data
model = keras.Sequential([
    layers.Input(input_shape),
    layers.BatchNormalization(),
    layers.Dense(512, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dense(512, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dense(512, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dense(1),
])
model.compile(
    optimizer = "sgd", # SGD is more sensitive to differences of scale
    loss = "mae",
    metrics = ["mae"],
)
EPOCHS = 100
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 64,
    epochs = EPOCHS,
    verbose = 0,
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
ax.set_title("Concrete Model with Batch Normalization")
fname = "results/Dropout_Batch_Normalization_Exercise_ConcreteModel_BatchNorm"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

"""
4) Evaluate Batch Normalization

Did adding batch normalization help?
    You can see that adding batch normalization was a big improvement on the first attempt!
    By adaptively scaling the data as it passes through the network, batch normalization can let you train models on difficult datasets.
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
