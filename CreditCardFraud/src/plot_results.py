import time
tic = time.time()

import argparse
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            value = value.split(",")
            getattr(namespace, self.dest)[key] = value

parser = argparse.ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("res_type", choices=["Train", "Test"], help="Choose to plot results from model training or testing.")
parser.add_argument("--grouping", nargs="*", action=ParseKwargs, help="Specifies how results should be grouped for plotting.")
args = parser.parse_args()

from pdb import set_trace
import os
import pandas as pd
# plotting styles
import utils.plotting_scripts as plt_scripts

## check if output directory exists and make it if it doesn't
resdir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir)
if not os.path.isdir(resdir):
    raise ValueError(f"{residr} could not be found")
output_dir = os.path.join(resdir, f"{args.res_type}ing")

# open file which has traiing results
import pickle
input_fname = os.path.join(resdir, f"{args.res_type}ingResults.pkl")
if not os.path.isfile(input_fname): raise ValueError(f"Could not find {input_fname}.")
try:
    results_dict = pickle.load(open(input_fname, "rb"))
except:
    raise ValueError(f"Could not open {input_fname}.")

results_df = pd.DataFrame(results_dict[f"{args.res_type}_Results"], columns=sorted(results_dict[f"{args.res_type}_Results"].keys()))
if args.grouping:
    if len(list(args.grouping.keys())) > 1: raise ValueError("Only grouping by 1 category is allowed.")
    group_cat = list(args.grouping.keys())[0]
    group_vals = args.grouping[group_cat]
    if len(group_vals) > 1: raise ValueError("Only grouping by 1 value is allowed.")
    group_val = group_vals[0]

    import itertools
    classifiers = sorted(set([col.split(" ")[0] for col in results_df.columns]))
    samplings = sorted(set([" ".join(col.split(" ")[1:-1]) for col in results_df.columns]))
    par_optimization = sorted(set([col.split(" ")[-1] for col in results_df.columns]))
    if group_cat == "Classifier":
        classifiers = group_vals
    elif group_cat == "Sampling":
        samplings = group_vals
    elif group_cat == "Optimization":
        par_optimization = group_vals
    else: raise ValueError(f"Grouping category {group_cat} not allowed.")

    combinations = list(itertools.product(*[classifiers, samplings, par_optimization]))
    rename_dict = {" ".join(combo) : " ".join(" ".join(combo).replace(group_val, "").split()) for combo in combinations}
    results_df = results_df.rename(columns=rename_dict)[rename_dict.values()]

    output_dir = os.path.join(output_dir, f"{group_cat}_Grouping", "".join(group_val.split()))


## check if output directory exists and make it if it doesn't
if not os.path.isdir(output_dir): os.makedirs(output_dir)

results_cols = ["Cross_Val", "Precision", "Recall", "F1"] if args.res_type == "Train" else ["Precision", "Recall", "F1"]
print(results_df.loc[results_cols, :].transpose())

# plot metric results
fig = plt_scripts.plot_df(results_df.loc[results_cols, :], fig_title=f"{args.res_type}ing Results ({group_val} Models)" if args.grouping else f"{args.res_type}ing Results")
fname = os.path.join(output_dir, f"{args.res_type}ing_Results_Table_{''.join(group_val.split())}Models" if args.grouping else f"{args.res_type}ing_Results_Table")
fig.savefig(fname)
print(f"{fname} written")
fig.clear()

# plot confusion matrices
fig = plt_scripts.plot_confusion_matrix(df=results_df.loc[["Confusion_Matrix"], :].transpose(),
        fig_title=f"Confusion Matrix for {args.res_type}ing Data ({group_val} Models)" if args.grouping else f"Confusion Matrix for {args.res_type}ing Data")
fname = os.path.join(output_dir, f"{args.res_type}ing_ConfusionMatrix_{''.join(group_val.split())}Models" if args.grouping else f"{args.res_type}ing_ConfusionMatrix")
fig.savefig(fname)
print(f"{fname} written")
fig.clear()

# plot ROC curves for testing dataset
fig = plt_scripts.plot_roc(df=results_df.loc[[col for col in results_df.index if "ROC" in col], :].transpose(),
        fig_title=f"ROC Curves ({group_val} Models)" if args.grouping else "ROC Curves")
fname = os.path.join(output_dir, f"{args.res_type}ing_ROC_AUC_{''.join(group_val.split())}Models" if args.grouping else f"{args.res_type}ing_ROC_AUC")
fig.savefig(fname)
print(f"{fname} written")
fig.clear()

    # make ROC curve using only false positive rates < 0.1%
fpr_thresh = 0.001
fig = plt_scripts.plot_roc(df=results_df.loc[[col for col in results_df.index if "ROC" in col], :].transpose(), fpr_thresh=fpr_thresh,
        fig_title=f"ROC Curves ({group_val} Models)" if args.grouping else "ROC Curves")
fname = os.path.join(output_dir, f"{args.res_type}ing_ROC_AUC_{str(fpr_thresh).replace('.', 'p')}_{''.join(group_val.split())}Models" if args.grouping\
        else f"{args.res_type}ing_ROC_AUC_{str(fpr_thresh).replace('.', 'p')}")
fig.savefig(fname)
print(f"{fname} written")
fig.clear()

# plot precision-recall curves for testing dataset
fig = plt_scripts.plot_precision_recall(df=results_df.loc[[col for col in results_df.index if "PRCurve" in col], :].transpose(),
        fig_title=f"Precision-Recall Curve ({group_val} Models)" if args.grouping else "Precision-Recall Curve")
fname = os.path.join(output_dir, f"{args.res_type}ing_PrecisionRecall_AUC_{''.join(group_val.split())}Models" if args.grouping\
        else f"{args.res_type}ing_PrecisionRecall_AUC")
fig.savefig(fname)
print(f"{fname} written")
fig.clear()

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
