import time
tic = time.time()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("res_type", choices=["Train", "Test"], help="Choose to plot results from model training or testing.")
args = parser.parse_args()

from pdb import set_trace
import os

# plotting styles
import utils.plotting_scripts as plt_scripts

import pandas as pd

## check if output directory exists and make it if it doesn't
resdir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir)
if not os.path.isdir(resdir):
    raise ValueError(f"{residr} could not be found. Please check if input is correct, otherwise run 'preprocessing.py'.")
output_dir = os.path.join(resdir, f"{args.res_type}ing")
if not os.path.isdir(output_dir): os.makedirs(output_dir)

# open file which has traiing results
import pickle
input_fname = os.path.join(resdir, f"{args.res_type}ingResults.pkl")
if not os.path.isfile(input_fname): raise ValueError(f"Could not find {input_fname}.")
try:
    results_dict = pickle.load(open(input_fname, "rb"))
except:
    raise ValueError(f"Could not open {input_fname}.")

scale = results_dict["MetaData"]["Scaler"]


if args.res_type == "Train":
    # plot metric results
    results_df = pd.DataFrame(results_dict[f"{args.res_type}_Results"])
    fig = plt_scripts.plot_df(results_df.loc[["Cross_Val", "Precision", "Recall", "F1"], :], data_type=f"{args.res_type}ing")
    fname = os.path.join(output_dir, f"{args.res_type}ing_Results_Table_{scale}Scaler")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()

    # plot confusion matrices
    fig = plt_scripts.plot_confusion_matrix(df=results_df.loc[["Confusion_Matrix"], :].transpose(), data_type="Testing")
    fname = os.path.join(output_dir, f"Testing_ConfusionMatrix_{scale}Scaler")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()



if args.res_type == "Test":
    # plot metric results
    results_df = pd.DataFrame(results_dict[f"{args.res_type}_Results"])
    #fig = plt_scripts.plot_pandas_df(pd.DataFrame(results), data_type=f"{args.res_type}ing")
    fig = plt_scripts.plot_df(results_df.loc[["Precision", "Recall", "F1"], :], data_type=f"{args.res_type}ing")
    fname = os.path.join(output_dir, f"{args.res_type}ing_Results_Table_{scale}Scaler")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()

    # plot confusion matrices
    fig = plt_scripts.plot_confusion_matrix(df=results_df.loc[["Confusion_Matrix"], :].transpose(), data_type="Testing")
    fname = os.path.join(output_dir, f"Testing_ConfusionMatrix_{scale}Scaler")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()

    # plot ROC curves for testing dataset
    fig = plt_scripts.plot_roc(df=results_df.loc[[col for col in results_df.index if "ROC" in col], :].transpose())
    fname = os.path.join(output_dir, f"Testing_ROC_AUC_{scale}Scaler")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()

        # make ROC curve using only false positive rates < 0.1%
    fpr_thresh = 0.001
    fig = plt_scripts.plot_roc(df=results_df.loc[[col for col in results_df.index if "ROC" in col], :].transpose(), fpr_thresh=fpr_thresh)
    fname = os.path.join(output_dir, f"Testing_ROC_AUC_{scale}Scaler_{str(fpr_thresh).replace('.', 'p')}")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()

    #set_trace()

    # plot precision-recall curves for testing dataset
    #fig = plt_scripts.plot_precision_recall(X=features, y=target, classifiers=trained_models_dict["Models"])
    fig = plt_scripts.plot_precision_recall(df=results_df.loc[[col for col in results_df.index if "PRCurve" in col], :].transpose())
    fname = os.path.join(output_dir, f"Testing_PrecisionRecall_AUC_{scale}Scaler")
    fig.savefig(fname)
    print(f"{fname} written")
    fig.clear()


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
