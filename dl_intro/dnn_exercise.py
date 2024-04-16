import time
tic = time.time()

"""
Introduction

In the tutorial, we saw how to build deep neural networks by stacking layers inside a Sequential model.
By adding an activation function after the hidden layers, we gave the network the ability to learn more complex (non-linear) relationships in the data.

In these exercises, you'll build a neural network with several hidden layers and then explore some activation functions beyond ReLU. Run this next cell to set everything up!
"""

from pdb import set_trace

import pandas as pd
import utils.Utilities as Utils

concrete = Utils.csv_to_pandas_DF("concrete.csv")
print(concrete.head())

"""
1) Input Shape

The target for this task is the column 'CompressiveStrength'. The remaining columns are the features we'll use as inputs.

What would be the input shape for this dataset?
    concrete.shape = (1030, 9) -> input_shape = [concrete.shape[1] - 1]
"""

input_shape = [concrete.shape[1] - 1]

"""
2) Define a Model with Hidden Layers

Now create a model with three hidden layers, each having 512 units and the ReLU activation.
Be sure to include an output layer of one unit and no activation, and also input_shape as an argument to the first layer.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units = 512, activation = "relu", input_shape = input_shape),
    layers.Dense(units = 512, activation = "relu"),
    layers.Dense(units = 512, activation = "relu"),
    layers.Dense(units = 1)
])

"""
3) Activation Layers

Let's explore activations functions some.

The usual way of attaching an activation function to a Dense layer is to include it as part of the definition with the activation argument.
Sometimes though you'll want to put some other layer between the Dense layer and its activation function.
(We'll see an example of this in Lesson 5 with batch normalization.)
In this case, we can define the activation in its own Activation layer, like so:

    layers.Dense(units=8),
    layers.Activation('relu')

This is completely equivalent to the ordinary way: layers.Dense(units=8, activation='relu').

Rewrite the following model so that each activation is in its own Activation layer.

### YOUR CODE HERE: rewrite this to use activation layers
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[8]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
"""

### YOUR CODE HERE: rewrite this to use activation layers
model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation("relu"),
    layers.Dense(32),
    layers.Activation("relu"),
    layers.Dense(1),
])

"""
Alternatives to ReLU

There is a whole family of variants of the 'relu' activation -- 'elu', 'selu', and 'swish', among others -- all of which you can use in Keras.
Sometimes one activation will perform better than another on a given task, so you could consider experimenting with activations as you develop a model.
The ReLU activation tends to do well on most problems, so it's a good one to start with.

Let's look at the graphs of some of these. Change the activation from 'relu' to one of the others named above.
Then run the cell to see the graph. (Check out the documentation for more ideas.): https://www.tensorflow.org/api_docs/python/tf/keras/activations
"""

# Change 'relu' to 'elu', 'selu', 'swish'... or something else
layer_opts = ["relu", "elu", "selu", "swish"]

# Setup plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)

fig, ax = plt.subplots(dpi = 100)
x = tf.linspace(-3.0, 3.0, 100)

for layer in layer_opts:
    activation_layer = layers.Activation(layer)
    y = activation_layer(x)
    ax.plot(x, y, label=layer)

ax.autoscale()
ax.set_xlim(-3., 3.)
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.legend()
ax.set_title("Activation Functions")
fname = "results/DNN_ActivationFunctions"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
