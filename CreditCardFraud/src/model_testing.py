import time, datetime
tic = time.time()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("--debug", action="store_true", help="Output all print statements for debugging purposes.")
parser.add_argument("--no_results", action="store_true", help="Suppress making output file.")
args = parser.parse_args()

from pdb import set_trace
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score

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

scale = metadata["MetaData"]["Scaler"]
# set random seeds for reproducibility
rand_state = metadata["MetaData"]["Random_State"]
np.random.seed(rand_state)


# test models on testing data
def test_models(X, y, classifiers=dict()):
    ## fit and evaluate each classifier of the testing set
    results = {}
    for key, classifier in classifiers.items():
        print(f"\n\tTesting {key}...")
        y_pred, y_pred_prob = classifier.predict(X), classifier.predict_proba(X)[:, 1] # index 1 is chosen because we're intersted in the 'fraud' class label

        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")

            # save quantities useful for plotting ROC curves, confusion matrices
        fpr_array, tpr_array, roc_thresh = roc_curve(y, y_pred_prob)
        precision_array, recall_array, pr_thresh = precision_recall_curve(y, y_pred_prob)

        results[key] = {
            "Precision" : precision, "Recall" : recall, "F1" : f1,
            "ROC_FPR" : fpr_array, "ROC_TPR" : tpr_array, "ROC_Thresh" : roc_thresh,
            "Confusion_Matrix" : confusion_matrix(y, y_pred),
            "PRCurve_Precision" : precision_array, "PRCurve_Recall" : recall_array, "PRCurve_Thresh" : pr_thresh, "PRCurve_AvgPrec" : average_precision_score(y, y_pred_prob),
        }
        print(f"\tFinished testing {key}.")

    return results


# Test each of the models using the testing data
results_dict = test_models(X=features, y=target, classifiers=trained_models_dict["Models"])
print(f"\n\n---------- Model Testing Completed ----------\n\n{pd.DataFrame(results_dict, columns=list(results_dict.keys()), index=['Precision', 'Recall', 'F1']).transpose()}")

# save results as pickle file
if not args.no_results:
    outfname = os.path.join(resdir, "TestingResults.pkl")
    outdict = {
        "MetaData" : metadata,
        "Test_Results"  : results_dict,
    }
    
    with open(outfname, "wb") as outfile:
        pickle.dump(outdict, outfile)
    print(f"{outfname} written")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
