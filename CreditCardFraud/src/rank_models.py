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
parser.add_argument("--grouping", nargs="*", action=ParseKwargs, help="Specifies how results should be grouped for plotting.")
args = parser.parse_args()

from pdb import set_trace
import os
import pandas as pd
from utils.compile_metrics import get_roc_auc

## check if output directory exists and make it if it doesn't
input_dir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir)
if not os.path.isdir(input_dir):
    raise ValueError(f"{input_dir} could not be found")

# open file which has traiing results
import pickle
input_fname = os.path.join(input_dir, "TestingResults.pkl")
if not os.path.isfile(input_fname): raise ValueError(f"Could not find {input_fname}.")
try:
    results_dict = pickle.load(open(input_fname, "rb"))
except:
    raise ValueError(f"Could not open {input_fname}.")

# create pandas dataframe
results_df = pd.DataFrame(results_dict["Test_Results"], columns=sorted(results_dict["Test_Results"].keys()))
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

results_df = results_df.transpose()

# add ROC AUC to results_df
results_df["ROC_AUC"] = results_df.apply(lambda x: get_roc_auc(x["ROC_FPR"], x["ROC_TPR"]), axis=1)
results_df["ROC_AUC_0p001"] = results_df.apply(lambda x: get_roc_auc(x["ROC_FPR"], x["ROC_TPR"], fpr_thresh=0.001), axis=1)

# calculate rankings for each model based for different metrics
metrics_to_sort_by = {"F1" : "F1", "PRCurve_AvgPrec" : "PR AP", "ROC_AUC_0p001" : "ROC AUC 0.001"}
rankings_df = results_df.copy()
for metric, metric_name in metrics_to_sort_by.items():
    rankings_df = rankings_df.sort_values(metric, axis=0, ascending=False)
    rankings_df[f"{metric_name} Rank"] = range(1, len(rankings_df) + 1)

# only keep ranking columns
rankings_df.drop([col for col in rankings_df.columns if "Rank" not in col], axis=1, inplace=True)

# calculate average ranking
rankings_df["Average Rank"] = rankings_df.mean(axis=1)
rankings_df = rankings_df.sort_values("Average Rank")
print(rankings_df)

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
