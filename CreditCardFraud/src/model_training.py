import time
tic = time.time()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("--no_optimize", action="store_true", help="Ignore all parameter optimization algorithms and just train default models.")
parser.add_argument("--debug", action="store_true", help="Output all print statements for debugging purposes.")
parser.add_argument("--no_results", action="store_true", help="Suppress making output file.")
args = parser.parse_args()

from pdb import set_trace
import os
import pandas as pd
import numpy as np
from copy import deepcopy

import utils.model_options as model_opts
from sklearn.base import clone # needed for 'initializing' models for each resampling method
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
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
        if (metadata["Optimizers"] != ["Default"]) and (not args.no_optimize):
        #if args.optimize:
            test_features, test_target = storedata["testing_features"].copy(), storedata["testing_target"].copy()
    storedata.close()
except:
    print(f"Could not get data from {inputfname}")

scale = metadata["MetaData"]["Scaler"]

# set random seeds for reproducibility
rand_state = metadata["MetaData"]["Random_State"]
np.random.seed(rand_state)


# get optimizer info
optimizer_algos = metadata["Optimizers"]
if args.no_optimize: optimizer_algos = ["Default"]
if optimizer_algos != ["Default"]:
    import utils.model_optimization as optimize_model
        ## plotting styles
    import utils.plotting_scripts as plt_scripts
    opt_dir = os.path.join(resdir, "ParameterOptimization")
    for algo in optimizer_algos:
        if algo == "Default": continue
        if not os.path.isdir(os.path.join(opt_dir, algo)): os.makedirs(os.path.join(opt_dir, algo))


# fit and evaluate models on training data
def fit_train_models(X, y, classifiers=dict()):
    ## fit and evaluate each classifier of the training set
    results = {}

    for key, classifier in classifiers.items():
        print(f"\n\tTraining {key}...")
        classifier.fit(X, y)
        cross_val_scores = cross_val_score(classifier, X, y, cv=5)
        precision, recall, f1, _ = precision_recall_fscore_support(y, classifier.predict(X), average="binary")

        results[key] = {
            "Precision" : precision, "Recall" : recall, "F1" : f1, "Cross_Val" : cross_val_scores.mean(),
            "Confusion_Matrix" : confusion_matrix(y, classifier.predict(X)),
        }

        print(f"\tFinished training {key}.")

    return results, classifiers


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

classifiers_dict, results_dict = {}, {}


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

    #set_trace()
    tmp_classifiers_dict = {}
    for optimizer_algo in optimizer_algos:
        print(f"\n\t{optimizer_algo} is being optimized")
        if optimizer_algo == "Default":
            tmp_classifiers_dict.update({f"{key} {pipe_name} {optimizer_algo}" :  clone(val) for key, val in classifiers_options.items()})
        else:
            tmp_classifiers_dict.update({f"{key} {pipe_name} {optimizer_algo}" :  clone(val) for key, val in classifiers_options.items()})
            for key, val in classifiers_options.items():
                clf = optimize_model.optimize_model(
                    classifier=val, algo=optimizer_algo,
                    data={"X_train" : X_res.copy(), "y_train" : y_res.copy(), "X_test" : test_features.copy(), "y_test" : test_target.copy()}
                )
                    # plot score for each hyperparameter combination
                fig = plt_scripts.plot_optimization_results(clf, class_type=f"{key} {pipe_name} {optimizer_algo}")
                if isinstance(fig, list):
                    for idx in range(len(fig)):
                        fname = os.path.join(opt_dir, optimizer_algo, f"{key}_{pipe_name.replace(' ', '_')}_{scale}Scaler_ValidationCurves_{optimizer_algo}_{idx}")
                        fig[idx].savefig(fname)
                        print(f"{fname} written")
                        fig[idx].clear()
                else:
                    fname = os.path.join(opt_dir, optimizer_algo, f"{key}_{pipe_name.replace(' ', '_')}_{scale}Scaler_ValidationCurves_{optimizer_algo}")
                    fig.savefig(fname)
                    print(f"{fname} written")
                    fig.clear()

                if optimizer_algo == "GASearchCV":
                        # plot fitness evolution
                    fig = plt_scripts.plot_GAsearch_results(clf, plot_type="FitnessEvolution")
                    fname = os.path.join(opt_dir, optimizer_algo, f"{key}_{pipe_name.replace(' ', '_')}_{scale}Scaler_FitnessEvolution_{optimizer_algo}")
                    fig.savefig(fname)
                    print(f"{fname} written")
                    fig.clear()

                        # plot parameter search space
                    fig = plt_scripts.plot_GAsearch_results(clf, plot_type="SearchSpace")
                    fname = os.path.join(opt_dir, optimizer_algo, f"{key}_{pipe_name.replace(' ', '_')}_{scale}Scaler_SearchSpace_{optimizer_algo}")
                    fig.savefig(fname)
                    print(f"{fname} written")
                    fig.clear()
                    #set_trace()

                tmp_classifiers_dict[f"{key} {pipe_name} {optimizer_algo}"] = clf.best_estimator_

    tmp_results_dict, tmp_classifiers_dict = fit_train_models(X=X_res, y=y_res, classifiers=tmp_classifiers_dict)
    classifiers_dict.update(tmp_classifiers_dict)

    results_dict.update(tmp_results_dict)


print(f"\n\n---------- Model Training Completed ----------\n\n{pd.DataFrame(results_dict, columns=list(results_dict.keys()), index=['Cross_Val', 'Precision', 'Recall', 'F1']).transpose()}")


# save results as pickle file
if not args.no_results:
    outfname = os.path.join(resdir, "TrainingResults.pkl")

    outdict = {
        "MetaData" : metadata,
        "Models" : classifiers_dict,
        "Train_Results" : results_dict,
    }
    
    import pickle
    with open(outfname, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print(f"{outfname} written")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
