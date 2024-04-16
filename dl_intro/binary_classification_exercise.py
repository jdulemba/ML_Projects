import time
tic = time.time()

"""
In this exercise, you'll build a model to predict hotel cancellations with a binary classifier.
"""

from pdb import set_trace
import utils.Utilities as Utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

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


## load and preprocess data
hotel = Utils.csv_to_pandas_DF("hotel.csv")

X = hotel.copy()
y = X.pop("is_canceled")

X["arrival_date_month"] = X["arrival_date_month"].map(
    {
        "January" : 1, "February" : 2, "March" : 3,
        "April" : 4, "May" : 5, "June" : 6,
        "July" : 7, "August" : 8, "September" : 9,
        "October" : 10, "November" : 11, "December" : 12
    }
)

features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]
features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, train_size = 0.75, random_state = 0)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

"""
1) Define Model

The model we'll use this time will have both batch normalization and dropout layers.
To ease reading we've broken the diagram into blocks, but you can define it layer by layer as usual.
"""

model = keras.Sequential([
    layers.Input([X_train.shape[1]]),
    layers.BatchNormalization(),
    layers.Dense(256, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation = "sigmoid"),
])

"""
2) Add Optimizer, Loss, and Metric

Now compile the model with the Adam optimizer and binary versions of the cross-entropy loss and accuracy metric.
"""

model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["binary_accuracy"],
)

"""Run to train the model and plot the learning curves."""

early_stopping = keras.callbacks.EarlyStopping(
    patience = 5,
    min_delta = 0.001,
    restore_best_weights = True,
)

history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 512,
    epochs = 200,
    callbacks = [early_stopping],
    verbose = 0, # hide the output because we have so many epochs
)
print(f"Best Validation Loss: {np.min(np.array(history.history['val_loss']))}")
print(f"Best Validation Accuracy: {np.min(np.array(history.history['val_binary_accuracy']))}\n")

## plot training and validation loss (cross-entropy) and accuracy
fig, ax = plt.subplots(2, 1, sharex=True, dpi = 100)
    ## cross-entropy
ax[0].plot(history.history["loss"], "k", label = "Training")
ax[0].plot(history.history["val_loss"], "b", label = "Validation")
ax[0].autoscale()
ax[0].set_xlim(history.epoch[0], history.epoch[-1])
ax[0].set_ylabel("Cross-Entropy")
ax[0].set_title("Model Cross-Entropy and Accuracy per Epoch")
ax[0].legend()

    ## accuracy
ax[1].plot(history.history["binary_accuracy"], "k", label = "Training")
ax[1].plot(history.history["val_binary_accuracy"], "b", label = "Validation")
ax[1].autoscale()
ax[1].set_ylabel("Accuracy")
ax[1].set_xlim(history.epoch[0], history.epoch[-1])
ax[1].set_xlabel("Epoch")

fname = "results/Binary_Classification_Exercise_Loss_and_Accuracy_Per_Epoch"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
3) Train and Evaluate

What do you think about the learning curves? Does it look like the model underfit or overfit?
Was the cross-entropy loss a good stand-in for accuracy?
    Though we can see the training loss continuing to fall, the early stopping callback prevented any overfitting. 
    Moreover, the accuracy rose at the same rate as the cross-entropy fell, so it appears that minimizing cross-entropy was a good stand-in.
    All in all, it looks like this training was a success!
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
