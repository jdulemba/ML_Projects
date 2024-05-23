import time
tic = time.time()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("--debug", action="store_true", help="Output all print statements for debugging purposes.")
parser.add_argument("--no_results", action="store_true", help="Suppress making output file.")
parser.add_argument("--no_plots", action="store_true", help="Suppress making all plots.")
args = parser.parse_args()

from pdb import set_trace
import os

# plotting styles
if not args.no_plots:
    import utils.plotting_scripts as plt_scripts

import pandas as pd
import numpy as np
from copy import deepcopy

import utils.model_options as model_opts

from sklearn.base import clone # needed for 'initializing' models for each resampling method
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

## check if output directory exists and make it if it doesn't
resdir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir)
if not os.path.isdir(resdir):
    raise ValueError(f"{residr} could not be found. Please check if input is correct, otherwise run 'preprocessing.py'.")

# getting data
inputfname = os.path.join(resdir, "processed_data.hdf5")
try:
    with pd.HDFStore(inputfname) as storedata:
        features, target = storedata["training_features"].copy(), storedata["training_target"].copy()
        metadata = storedata.get_storer("training_features").attrs.metadata
    storedata.close()
except:
    print(f"Could not get data from {inputfname}")

scale = metadata["Scaler"]

# set random seeds for reproducibility
rand_state = metadata["Random_State"]
np.random.seed(rand_state)


# fit and evaluate models on training data
def fit_train_models(X, y, classifiers=dict(), results_df=None):
    ## fit and evaluate each classifier of the training set
    if results_df is None:
        results_df = pd.DataFrame(index=list(classifiers.keys()), columns=["Cross_Val", "Precision", "Recall", "F1"])

    for key, classifier in classifiers.items():
        print(f"\n\tTraining {key}...")
        classifier.fit(X, y)
        cross_val_scores = cross_val_score(classifier, X, y, cv=5)
        precision, recall, f1, _ = precision_recall_fscore_support(y, classifier.predict(X), average="binary")
        results_df.loc[key, :] = np.array([cross_val_scores.mean(), precision, recall, f1])
        print(f"\tFinished training {key}.")

    # create and plot confustion matrix for each classifier
    if not args.no_plots:
        fig = plt_scripts.plot_confusion_matrix(X=X, y=y, classifiers=classifiers, data_type="Training")
        class_pipename = sorted(set(["_".join(key.split(" ")[1:]) for key in classifiers.keys()]))[0]
        fname = os.path.join(resdir, f"Training_ConfusionMatrix_{scale}Scaler_{class_pipename.replace(' ', '_')}")
        fig.savefig(fname, bbox_inches="tight")
        print(f"{fname} written")
        fig.clear()

    return results_df, classifiers




# define and make pipeline for resampling methods
sampling_method_opts = deepcopy(metadata["Sampling Methods"])
# add random state value into options dict
def update_dict(input_dict):
    for k, v in input_dict.copy().items():
        if isinstance(v, dict):     # For DICT
            if not v: # if dict is empty
                v["random_state"] = rand_state
            else:
                input_dict[k] = update_dict(v)
        else: # Update Key-Value
            input_dict["random_state"] = rand_state

    return input_dict

sampling_method_opts = update_dict(sampling_method_opts)
pipelines_dict = model_opts.pipeline_dict_constructor(**sampling_method_opts)

classifier_opts = deepcopy(metadata["Classifiers"])
classifier_opts = update_dict(classifier_opts)
classifiers_options = model_opts.classifiers_dict_constructor(**classifier_opts)

classifiers_results = {}
training_results_df = pd.DataFrame(columns=["Cross_Val", "Precision", "Recall", "F1"])


    # perform testing and analysis for each type of resampling strategy
for pipe_name, pipeline in pipelines_dict.items():
    print(f"\n\t\tResampling Scheme {pipe_name} is being evaluated now...")
    # perform resampling
        # for some reason this doesn't work when using only 1 line
    if pipeline is None:
        X_res, y_res = features.copy(), target.copy()
    else:
        X_res, y_res = pipeline.fit_resample(features.copy(), target.copy())

    # Confirm ratios
    if args.debug:
        print(f"Total number of transactions in resampled data ({pipe_name}): {len(y_res)}")
        print(f"Number of normal transactions: {np.sum(y_res == 0)} ({np.sum(y_res == 0)/len(y_res)}%)")
        print(f"Number of fraud transactions: {np.sum(y_res == 1)} ({np.sum(y_res == 1)/len(y_res)}%)")


        # create temp dict of classifiers to add to the results dict
    tmp_classifiers_dict = {f"{key} {pipe_name}" : clone(val) for key, val in classifiers_options.items()}
    training_results_df, tmp_classifiers_dict = fit_train_models(X=X_res, y=y_res, classifiers=tmp_classifiers_dict, results_df=training_results_df)
    classifiers_results.update(tmp_classifiers_dict)

print(f"\n\n---------- Model Training Completed ----------\n\n{training_results_df}")

# save training results as plots
if not args.no_plots:
    fig = plt_scripts.plot_pandas_df(training_results_df.transpose(), data_type="Training")
    fname = os.path.join(resdir, f"Training_Results_Table_{scale}Scaler.png")
    fig.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    fig.clear()


# save results as pickle file
if not args.no_results:
    outfname = os.path.join(resdir, "TrainingResults.pkl")
    outdict = {
        "MetaData" : metadata,
        "Models" : classifiers_results,
        "Train_Results" : training_results_df.to_dict(),
    }
    
    import pickle
    with open(outfname, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print(f"{outfname} written")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
