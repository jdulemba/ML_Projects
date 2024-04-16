import time
tic = time.time()

"""
In this exercise, you'll build a model to detect the Higgs Boson with a binary classifier.
"""

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("device", choices=["CPU", "GPU"], default="CPU", help="Choose to run model using CPU or GPU.")
parser.add_argument("--test", action="store_true", help="Only use 1 file for training and validation to test workflow.")
args = parser.parse_args()

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)

from pdb import set_trace
import utils.Utilities as Utils
import numpy as np
import pandas as pd

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
print("Tensorflow version " + tf.__version__)

# Model Configuration
UNITS = 2 ** 8 if args.test else 2 ** 11 # 2048
ACTIVATION = "relu"
DROPOUT = 0.1

# Training Configuration
BATCH_SIZE_PER_REPLICA = 2 ** 8 if args.test else 2 ** 11 # powers of 128 are best


# Detect and init the TPU
try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    if args.device == "CPU":
        tf.config.set_visible_devices([], "GPU")
    elif args.device == "GPU":
        gpu = len(tf.config.list_physical_devices("GPU")) > 0
        if gpu: 
            tf.device(tf.config.list_physical_devices("GPU")[0]) ## use GPU
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
    

# Load Data
from tensorflow.io import FixedLenFeature
AUTO = tf.data.experimental.AUTOTUNE

"""
Load Data

The dataset has been encoded in a binary file format called TFRecords. 
These two functions will parse the TFRecords and build a TensorFlow tf.data.Dataset object that we can use for training.
"""

def make_decoder(feature_description):
    def decoder(example):
        example = tf.io.parse_single_example(example, feature_description)
        features = tf.io.parse_tensor(example["features"], tf.float32)
        features = tf.reshape(features, [28])
        label = example["label"]

        return features, label

    return decoder

def load_dataset(filenames, decoder, ordered=False):
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = (
        tf.data
        .TFRecordDataset(filenames, num_parallel_reads=AUTO)
        .with_options(ignore_order)
        .map(decoder, AUTO)
    )
    
    return dataset


dataset_size = int(11e6)
validation_size = int(5e5)
training_size = dataset_size - validation_size

# For model.fit
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
steps_per_epoch = training_size // batch_size
validation_steps = validation_size // batch_size

# For model.compile
steps_per_execution = 256

feature_description = {
    "features": FixedLenFeature([], tf.string),
    "label": FixedLenFeature([], tf.float32),
}
decoder = make_decoder(feature_description)

data_dir = "data"
train_files = [os.path.join(data_dir, "training", "shard_00.tfrecord"), os.path.join(data_dir, "training", "shard_01.tfrecord")] if args.test \
        else tf.io.gfile.glob(data_dir + "/training" + "/*.tfrecord")
valid_files = [os.path.join(data_dir, "validation", "shard_00.tfrecord"), os.path.join(data_dir, "validation", "shard_01.tfrecord")] if args.test \
        else tf.io.gfile.glob(data_dir + "/validation" + "/*.tfrecord")

ds_train = load_dataset(train_files, decoder, ordered=False)
ds_train = (
    ds_train
    .cache()
    .repeat()
    .shuffle(2 ** 19)
    .batch(batch_size)
    .prefetch(AUTO)
)

ds_valid = load_dataset(valid_files, decoder, ordered=False)
ds_valid = (
    ds_valid
    .batch(batch_size)
    .cache()
    .prefetch(AUTO)
)


"""
1) Define Model

Defining the deep branch of the network using Keras's Functional API, which is a bit more flexible that the Sequential method used before.
"""

def dense_block(units, activation, dropout_rate, l1 = None, l2 = None):
    def make(inputs):
        x = layers.Dense(units)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        return x

    return make


with strategy.scope():
    # Wide Network
    wide = keras.experimental.LinearModel()

    # Deep Network
    inputs = keras.Input(shape=[28])
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(inputs)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    x = dense_block(UNITS, ACTIVATION, DROPOUT)(x)
    outputs = layers.Dense(1)(x)
    deep = keras.Model(inputs = inputs, outputs = outputs)
    
    # Wide and Deep Network
    wide_and_deep = keras.experimental.WideDeepModel(
        linear_model = wide,
        dnn_model = deep,
        activation = "sigmoid",
    )

wide_and_deep.compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    #optimizer = "adam",
    metrics = ["AUC", "binary_accuracy"],
    steps_per_execution = steps_per_execution,
)


"""
2) Training

Use the EarlyStopping callback as usual and also defined a learning rate schedule.
It's been found that gradually decreasing the learning rate over the course of training can improve performance (the weights "settle in" to a minimum). 
This schedule will multiply the learning rate by 0.2 if the validation loss didn't decrease after an epoch.
"""

early_stopping = callbacks.EarlyStopping(
    patience = 2,
    min_delta = 0.001,
    restore_best_weights = True,
)

lr_schedule = callbacks.ReduceLROnPlateau(
    patience = 0,
    factor = 0.2,
    min_lr = 0.001,
)

history = wide_and_deep.fit(
    ds_train,
    validation_data = ds_valid,
    epochs = 50,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    callbacks = [early_stopping, lr_schedule],
    #verbose = 0,
)

print(f"Best Validation Loss: {np.min(np.array(history.history['val_loss']))}")
print(f"Best Validation AUC: {np.max(np.array(history.history['val_auc']))}\n")

## plot training and validation cross-entropy loss and area under curve
fig, ax = plt.subplots(2, 1, sharex=True, dpi = 100)
    ## cross-entropy
ax[0].plot(history.history["loss"], "k", label = "Training")
ax[0].plot(history.history["val_loss"], "b", label = "Validation")
ax[0].autoscale()
ax[0].set_xlim(history.epoch[0], history.epoch[-1])
ax[0].set_ylabel("Cross-Entropy Loss")
ax[0].set_title("Model Cross-Entropy and AUC per Epoch")
ax[0].legend()

    ## AUC
ax[1].plot(history.history["auc"], "k", label = "Training")
ax[1].plot(history.history["val_auc"], "b", label = "Validation")
ax[1].autoscale()
ax[1].set_ylabel("AUC")
ax[1].set_xlim(history.epoch[0], history.epoch[-1])
ax[1].set_xlabel("Epoch")

fname = f"results/HiggsDetection_Loss_and_AUC_Per_Epoch_{args.device}"
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
