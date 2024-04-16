import time
tic = time.time()

from pdb import set_trace

"""
Introduction

In this exercise you'll train a neural network on the Fuel Economy dataset and then explore the effect of the learning rate and batch size on SGD.
In the Fuel Economy dataset your task is to predict the fuel economy of an automobile given features like its type of engine or the year it was made.

First load the dataset.
"""

# Setup plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)
plt.rc("animation", html="html5")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
import utils.Utilities as Utils

fuel = Utils.csv_to_pandas_DF("fuel.csv")
X = fuel.copy()
# Remove target
y = X.pop("FE")

preprocessor = make_column_transformer(
    (
        StandardScaler(),
        make_column_selector(dtype_include=np.number)
    ),
    (
        OneHotEncoder(sparse_output=False),
        make_column_selector(dtype_include=object)
    ),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=input_shape),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1),
])

"""
1) Add Loss and Optimizer

Before training the network we need to define the loss and optimizer we'll use.
Using the model's compile method, add the Adam optimizer and MAE loss.
"""

model.compile(
    optimizer = "adam",
    loss = "mae",
)


"""
2) Train Model

Once you've defined the model and compiled it with a loss and optimizer you're ready for training.
Train the network for 200 epochs with a batch size of 128. The input data is X with target y.
"""

history = model.fit(
    X, y,
    batch_size = 128,
    epochs = 200,
)

"""The last step is to look at the loss curves and evaluate the training. Run the cell below to get a plot of the training loss."""


fig, ax = plt.subplots(dpi = 100)
ax.plot(history.history["loss"], "k")
ax.autoscale()
ax.set_xlim(history.epoch[0], history.epoch[-1])
ax.set_ylim(0, ax.get_ylim()[-1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Fuel Efficiency Model Loss per Epoch")
fname = "results/SGD_Exercise_Loss_Per_Epoch"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


"""
3) Evaluate Training

If you trained the model longer, would you expect the loss to decrease further?
    This depends on how the loss has evolved during training: if the learning curves have levelled off, there won't usually be any advantage to training for additional epochs.
    Conversely, if the loss appears to still be decreasing, then training for longer could be advantageous.

With the learning rate and the batch size, you have some control over:

    1) How long it takes to train a model
    2) How noisy the learning curves are
    3) How small the loss becomes
    
To get a better understanding of these two parameters, we'll look at the linear model, our simplest neural network.
Having only a single weight and a bias, it's easier to see what effect a change of parameter has.

The next cell (in the notebook) will generate an animation like the one in the tutorial.
Change the values for learning_rate, batch_size, and num_examples (how many data points) and then run the cell.
(It may take a moment or two.) Try the following combinations, or try some of your own:

learning_rate	batch_size	num_examples
    0.05	        32	        256
    0.05	        2	        256
    0.05	        128	        256
    0.02	        32	        256
    0.2	            32	        256
    1.0	            32	        256
    0.9	            4096	    8192
    0.99	        4096	    8192



4) Learning Rate and Batch Size

What effect did changing these parameters have?

You probably saw that smaller batch sizes gave noisier weight updates and loss curves.
This is because each batch is a small sample of data and smaller samples tend to give noisier estimates.
Smaller batches can have an "averaging" effect though which can be beneficial.

Smaller learning rates make the updates smaller and the training takes longer to converge.
Large learning rates can speed up training, but don't "settle in" to a minimum as well.
When the learning rate is too large, the training can fail completely. (Try setting the learning rate to a large value like 0.99 to see this.)
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
