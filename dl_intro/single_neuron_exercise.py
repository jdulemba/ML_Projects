import time
tic = time.time()

"""
Now for the exercise

The Red Wine Quality dataset consists of physiochemical measurements from about 1600 Portuguese red wines.
Also included is a quality rating for each wine from blind taste-tests.
"""

from pdb import set_trace

# Setup plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)

import pandas as pd
import utils.Utilities as Utils

red_wine = Utils.csv_to_pandas_DF("red-wine.csv")
#set_trace()
print(red_wine.head())
print(red_wine.shape) # (rows, columns) == (1599, 12)


"""
1) Input shape

How well can we predict a wine's perceived quality from the physiochemical measurements?

The target is 'quality', and the remaining columns are the features.i
How would you set the input_shape parameter for a Keras model on this task?

input_shape = [11]

2) Define a linear model

Now define a linear model appropriate for this task. Pay attention to how many inputs and outputs the model should have.
"""

from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
lin_unit = 1
model = keras.Sequential([
    layers.Dense(units = lin_unit, input_shape=[red_wine.shape[1] - lin_unit])
])

"""
3) Look at the weights

Internally, Keras represents the weights of a neural network with tensors.
Tensors are basically TensorFlow's version of a Numpy array with a few differences that make them better suited to deep learning.
One of the most important is that tensors are compatible with GPU and TPU) accelerators.
TPUs, in fact, are designed specifically for tensor computations.

A model's weights are kept in its weights attribute as a list of tensors. Get the weights of the model you defined above.
(If you want, you could display the weights with something like: print("Weights\n{}\n\nBias\n{}".format(w, b))).
"""

wval, bval = model.weights

"""
(By the way, Keras represents weights as tensors, but also uses tensors to represent data.
When you set the input_shape argument, you are telling Keras the dimensions of the array it should expect for each example in the training data.
Setting input_shape=[3] would create a network accepting vectors of length 3, like [0.2, 0.4, 0.6].)
"""


"""
Plot the output of an untrained linear model

The kinds of problems we'll work on through Lesson 5 will be regression problems, where the goal is to predict some numeric target.
Regression problems are like "curve-fitting" problems: we're trying to find a curve that best fits the data.
Let's take a look at the "curve" produced by a linear model. (You've probably guessed that it's a line!)

We mentioned that before training a model's weights are set randomly.
Run the cell below a few times to see the different lines produced with a random initialization.
(There's no coding for this exercise -- it's just a demonstration.)
"""


import tensorflow as tf

model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)
wval, bval = model.weights # you could also use model.get_weights() here

fig, ax = plt.subplots(dpi = 100)
ax.plot(x, y, "k")
ax.set_xlim(-1., 1.)
ax.set_ylim(-1., 1.)
ax.set_xlabel("Input: x")
ax.set_ylabel("Target y")
ax.set_title("Weight: {:0.2f}\nBias: {:0.2f}".format(wval[0][0], bval[0]))
fname = "results/SingleNeuron_Test"
fig.savefig(fname, bbox_inches="tight")
print(f"{fname} written")
plt.close(fig)

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
