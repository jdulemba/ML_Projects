import time, datetime
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

from sklearn.metrics import precision_recall_fscore_support

## check if output directory exists and make it if it doesn't
res_dir = os.environ["RESULTS_DIR"]

resdir = os.path.join(res_dir, args.jobdir)
if not os.path.isdir(resdir):
    raise ValueError(f"{residr} could not be found. Please check if input is correct, otherwise run 'preprocessing.py'.")


# open file which has traiing results
import pickle
train_fname = os.path.join(resdir, "TrainingResults.pkl")
if not os.path.isfile(train_fname): raise ValueError(f"Could not find {train_fname}.")
try:
    trained_models_dict = pickle.load(open(train_fname, "rb"))
except:
    raise ValueError(f"Could not open {train_fname}.")

metadata = trained_models_dict["MetaData"]

# getting data
inputfname = metadata["DataFname"]
try:
    with pd.HDFStore(inputfname) as storedata:
        features, target = storedata["testing_features"].copy(), storedata["testing_target"].copy()
    storedata.close()
except:
    print(f"Could not get data from {inputfname}")

scale = metadata["Scaler"]
# set random seeds for reproducibility
rand_state = metadata["Random_State"]
np.random.seed(rand_state)

# determine how many cases are fraud/not fraud
class_names = { 0 : "Not Fraud", 1 : "Fraud"}


# test models on testing data
def test_models(X, y, classifiers=dict()):
    ## fit and evaluate each classifier of the testing set
    results_df = pd.DataFrame(index=list(classifiers.keys()), columns=["Precision", "Recall", "F1"])
    for key, classifier in classifiers.items():
        print(f"\n\tTesting {key}...")
        precision, recall, f1, _ = precision_recall_fscore_support(y, classifier.predict(X), average="binary")
        results_df.loc[key, :] = np.array([precision, recall, f1])
        print(f"\tFinished testing {key}.")

        # create and plot confustion matrix for each classifier
    if not args.no_plots:
        fig = plt_scripts.plot_confusion_matrix(X=X, y=y, classifiers=classifiers, data_type="Testing")
        fname = os.path.join(resdir, f"Testing_ConfusionMatrix_{scale}Scaler")
        fig.savefig(fname, bbox_inches="tight")
        print(f"{fname} written")
        fig.clear()

    return results_df




# Test each of the models using the testing data
testing_results_df = test_models(X=features, y=target, classifiers=trained_models_dict["Models"])
print(f"\n\n---------- Model Testing Completed ----------\n\n{testing_results_df}")

# save testing results as plots
if not args.no_plots:
    fig = plt_scripts.plot_pandas_df(testing_results_df.transpose(), data_type="Testing")
    fname = os.path.join(resdir, f"Testing_Results_Table_{scale}Scaler")
    fig.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    fig.clear()

    # plot ROC curves for testing dataset
    fig = plt_scripts.plot_roc(X=features, y=target, classifiers=trained_models_dict["Models"])
    fname = os.path.join(resdir, f"Testing_ROC_AUC_{scale}Scaler")
    fig.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    fig.clear()

        # make ROC curve using only false positive rates < 0.1%
    fpr_thresh = 0.001
    fig = plt_scripts.plot_roc(X=features, y=target, classifiers=trained_models_dict["Models"], fpr_thresh=fpr_thresh)
    fname = os.path.join(resdir, f"Testing_ROC_AUC_{scale}Scaler_{str(fpr_thresh).replace('.', 'p')}")
    fig.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    fig.clear()


    # plot precision-recall curves for testing dataset
    fig = plt_scripts.plot_precision_recall(X=features, y=target, classifiers=trained_models_dict["Models"])
    fname = os.path.join(resdir, f"Testing_PrecisionRecall_AUC_{scale}Scaler")
    fig.savefig(fname, bbox_inches="tight")
    print(f"{fname} written")
    fig.clear()



# save results as pickle file
if not args.no_results:
    outfname = os.path.join(resdir, "TestingResults.pkl")
    outdict = {
        "MetaData" : metadata,
        "Test_Results"  : testing_results_df.to_dict(),
    }
    
    import pickle
    with open(outfname, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print(f"{outfname} written")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
