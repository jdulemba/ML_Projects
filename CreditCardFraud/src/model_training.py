import time
tic = time.time()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("--optimize", action="store_true", help="Output all print statements for debugging purposes.")
parser.add_argument("--debug", action="store_true", help="Output all print statements for debugging purposes.")
parser.add_argument("--no_results", action="store_true", help="Suppress making output file.")
args = parser.parse_args()

from pdb import set_trace
import os
import pandas as pd
import numpy as np
from copy import deepcopy

import utils.model_options as model_opts
if args.optimize:
    import utils.model_optimization as optimize_model
        ## plotting styles
    import utils.plotting_scripts as plt_scripts

from sklearn.base import clone # needed for 'initializing' models for each resampling method
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score

## check if output directory exists and make it if it doesn't
resdir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir)
if not os.path.isdir(resdir):
    raise ValueError(f"{residr} could not be found. Please check if input is correct, otherwise run 'preprocessing.py'.")
train_dir = os.path.join(resdir, "Training")
if not os.path.isdir(train_dir): os.makedirs(train_dir)
opt_dir = os.path.join(resdir, "ParameterOptimization")
if not os.path.isdir(opt_dir): os.makedirs(opt_dir)

# getting data
inputfname = os.path.join(resdir, "processed_data.hdf5")
try:
    with pd.HDFStore(inputfname) as storedata:
        features, target = storedata["training_features"].copy(), storedata["training_target"].copy()
        if args.optimize:
            test_features, test_target = storedata["testing_features"].copy(), storedata["testing_target"].copy()
        metadata = storedata.get_storer("training_features").attrs.metadata
    storedata.close()
except:
    print(f"Could not get data from {inputfname}")

scale = metadata["Scaler"]
optimizer_algo = metadata["Optimizer"]

# set random seeds for reproducibility
rand_state = metadata["Random_State"]
np.random.seed(rand_state)


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


    if args.optimize:
            # create temp dict of classifiers to add to the results dict
        tmp_classifiers_dict = {f"{key} {pipe_name} {optimizer_algo}" : None for key in classifiers_options.keys()}
        #set_trace()
        for key, val in classifiers_options.items():
            clf = optimize_model.optimize_model(
                classifier=val, algo=optimizer_algo,
                data={"X_train" : X_res.copy(), "y_train" : y_res.copy(), "X_test" : test_features.copy(), "y_test" : test_target.copy()}
            )
                # plot score for each hyperparameter combination
            fig = plt_scripts.plot_gridsearch_results(clf, class_type=f"{key} {pipe_name} {optimizer_algo}")
            fname = os.path.join(opt_dir, f"{key}_{pipe_name.replace(' ', '_')}_{scale}Scaler_ValidationCurves_ModelOptimized{optimizer_algo}")
            fig.savefig(fname)
            print(f"{fname} written")
            fig.clear()

            tmp_classifiers_dict[f"{key} {pipe_name} {optimizer_algo}"] = clf.best_estimator_

    else: # use default classifiers
        tmp_classifiers_dict = {f"{key} {pipe_name} Default" : clone(val) for key, val in classifiers_options.items()}
    tmp_results_dict, tmp_classifiers_dict = fit_train_models(X=X_res, y=y_res, classifiers=tmp_classifiers_dict)
    classifiers_dict.update(tmp_classifiers_dict)
    results_dict.update(tmp_results_dict)


print(f"\n\n---------- Model Training Completed ----------\n\n{pd.DataFrame(results_dict, columns=list(results_dict.keys()), index=['Cross_Val', 'Precision', 'Recall', 'F1']).transpose()}")


# save results as pickle file
if not args.no_results:
    outfname = os.path.join(resdir, "TrainingResults_DefaultModels.pkl") if not args.optimize else\
        os.path.join(resdir, f"TrainingResults_ModelOptimized{optimizer_algo}.pkl")

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
