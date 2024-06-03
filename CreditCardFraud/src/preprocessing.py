import time, datetime
tic = time.time()

"""
This file:
1. makes a plot showing how many fraud/normal transactions are present in the input csv file
2. scales features from the input data and makes a before vs after plot of the transformation (and only keeps the scaled data)
3. splits the data into training and testing sets
4. saves the split data and input parameters into a hdf5 file

The random state value, type of scaling to use, and train/test fraction value are all specified using the input cfile,
and subsequently stored in the output file.
"""

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("--cfile", default="default_config.json", help="Choose the config file from where to find the configuartion parameters.")
parser.add_argument("--debug", action="store_true", help="Output all print statements for debugging purposes.")
parser.add_argument("--no_results", action="store_true", help="Suppress making output file.")
parser.add_argument("--no_plots", action="store_true", help="Suppress making all plots.")
args = parser.parse_args()

from pdb import set_trace
import os
import json
from copy import deepcopy

# plotting styles
if not args.no_plots:
    # Set Matplotlib defaults
    from matplotlib import pyplot as plt
    from utils.styles import style_dict
    plt.style.use(style_dict)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

## check if output directory exists and make it if it doesn't
proj_dir = os.environ["PROJECT_DIR"]
res_dir = os.environ["RESULTS_DIR"]

resdir = os.path.join(res_dir, args.jobdir)
if not os.path.isdir(resdir): os.makedirs(resdir)
pltdir = os.path.join(resdir, "Preprocessing")
if not os.path.isdir(pltdir): os.makedirs(pltdir)

## get config parameters
cfile = os.path.join(proj_dir, "configs", args.cfile)
if not os.path.isfile(cfile): raise ValueError(f"Config file {cfile} not found.")
config = json.load(open(cfile))

# set random seeds for reproducibility
rand_state = config["MetaData"]["Random_State"]
np.random.seed(rand_state)


# open data
data = pd.read_csv(os.path.join(proj_dir, "data", "creditcard.csv"))

"""
data.columns
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')

There are 28 variables constructed through PCA, as well as Time, Amount, and Class (the target).
"""

# check if there are any null values
assert not data.isnull().any().any(), "There are unexpected Null values in the dataset"

# determine how many cases are fraud/not fraud
class_names = { 0 : "Not Fraud", 1 : "Fraud"}
target_name = "Class"

if not args.no_plots:
    count_classes = data[target_name].value_counts().rename(index = class_names).sort_index()
    # plot frequency of fraud/not fraud
    fig, ax = plt.subplots(constrained_layout=True)
    count_classes.plot(kind="bar", ax=ax)
    [ax.text(idx, val, str(val)+f" ({val/np.sum(count_classes.values)*100:.2f}%)", ha="center") for idx, val in enumerate(count_classes.values)]
    ax.set(title="Fraud Class Histogram", xlabel="Class", ylabel="Frequency")
    ax.set_yscale("log")
    
    fname = os.path.join(pltdir, "Preprocessing_Fraud_Frequency_Histogram")
    fig.savefig(fname)
    print(f"{fname} written")
    plt.close(fig)
    
    
    # plot each of the variables to view their distributions
    fig, ax = plt.subplots(5, 6, figsize = (15, 12), constrained_layout=True)
    fig.suptitle("Feature Distributions")
    
    for idx, feature in enumerate([col for col in data.columns if col != target_name]):
        ax.ravel()[idx].plot(data[feature])
        ax.ravel()[idx].set_xlabel(feature)
    
    fname = os.path.join(pltdir, "Preprocessing_Features_Distributions_vs_Index")
    fig.savefig(fname)
    print(f"{fname} written")
    plt.close(fig)

## Scale 'Time' and 'Amount' features, and compare distributions before/after scaling
scaler_opts_ = ["Standard", "Robust"]
scaler_type = config["MetaData"]["Scaler"]
if scaler_type == "Standard":
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
elif scaler_type == "Robust":
    ## RobustScaler is less prone to outliers.
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
else:
    raise IOError(f"Scaler type {scaler_type} is not a supported option.\nOnly {scaler_opts_} are currently available.")


data["scaled_amount"] = scaler.fit_transform(data["Amount"].values.reshape(-1,1))
data["scaled_time"] = scaler.fit_transform(data["Time"].values.reshape(-1,1))


if not args.no_plots:
    nbins = 20
    fig, ax = plt.subplots(2, 2, figsize=(18,4), constrained_layout=True)
        # plot original values
    ax[0, 0].hist(data["Amount"].values, bins=nbins, color="r", label="Original")
    ax[0, 0].set_title("Distribution of Transaction Amount")
    ax[0, 0].set_xlim([np.min(data["Amount"].values), np.max(data["Amount"].values)])
    ax[0, 0].set_yscale("log")
    ax[0, 0].legend()
    
    ax[0, 1].hist(data["Time"].values, bins=nbins, color="b", label="Original")
    ax[0, 1].set_title("Distribution of Transaction Time")
    ax[0, 1].set_xlim([np.min(data["Time"].values), np.max(data["Time"].values)])
    ax[0, 1].legend()
    
        # plot scaled values
    ax[1, 0].hist(data["scaled_amount"].values, bins=nbins, color="r", label=f"{scaler_type} Scaler")
    ax[1, 0].set_xlim([np.min(data["scaled_amount"].values), np.max(data["scaled_amount"].values)])
    ax[1, 0].set_yscale("log")
    ax[1, 0].legend()
    
    ax[1, 1].hist(data["scaled_time"].values, bins=nbins, color="b", label=f"{scaler_type} Scaler")
    ax[1, 1].set_xlim([np.min(data["scaled_time"].values), np.max(data["scaled_time"].values)])
    ax[1, 1].legend()
    
    fname = os.path.join(pltdir, f"Preprocessing_Time_Amount_Original_vs_{scaler_type}Scaler_Distributions")
    fig.savefig(fname)
    print(f"{fname} written")
    plt.close(fig)

# drop original features which were scaled
data = data.drop(["Time","Amount"], axis=1)

# determine features and target for entire dataset
features, target = data.iloc[:, data.columns != target_name], data[target_name]

# split into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target,
    train_size=config["MetaData"]["Training_Frac"], test_size=1-config["MetaData"]["Training_Frac"], random_state=rand_state
)

if args.debug:
    print(f"Number transactions train dataset: {len(features_train)}")
    print(f"Number transactions test dataset: {len(features_test)}")
    print(f"Total number of transactions: {len(features)}")



# store data in hdf5 file format
if not args.no_results:
    outfname = os.path.join(resdir, "processed_data.hdf5")
    storedata = pd.HDFStore(outfname)

    # store training and testing splits
    storedata.put("training_target", target_train)
    storedata.put("training_features", features_train)
    storedata.put("testing_target", target_test)
    storedata.put("testing_features", features_test)
     
    # store config parameters as metadata
    metadata = deepcopy(config)
    metadata["DataFname"] = outfname
    storedata.get_storer("training_features").attrs.metadata = metadata

    # closing the storedata
    storedata.close()
    print(f"{outfname} written")
     

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
