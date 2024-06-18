import time
tic = time.time()

import utils.model_options as model_opts
from pdb import set_trace

import argparse
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value

parser = argparse.ArgumentParser()
parser.add_argument("jobdir", help="The job directory name used as the output directory for this and the subsequent script's output.")
parser.add_argument("classifier", choices=model_opts.classifier_opts_, help="Choose the type of classifier to use.")
parser.add_argument("sampling", choices=model_opts.resampling_opts_, help="Choose the type of sampling technique to use")
parser.add_argument("optimizer", choices=["Default", "GridSearchCV", "RandomizedSearchCV", "GASearchCV"], help="Choose the hyperparameter optimization algorithm to use.")
parser.add_argument("--mode", choices=["Train", "Test", "Both"], default="Both", help="Choose whether to run the model traiing, testing, or both.")
parser.add_argument("--meta_info", nargs="*", action=ParseKwargs, help="Specify input parameters for metadata.")
parser.add_argument("--no_results", action="store_true", help="Suppress making output file.")
args = parser.parse_args()

import os
from copy import deepcopy
import pandas as pd
import numpy as np
import utils.analysis_steps as an_steps
from sklearn.base import clone # needed for 'initializing' models for each resampling method
import mlflow

key = " ".join([args.classifier, args.sampling, args.optimizer])

"""
retrieve data
"""
if (args.mode == "Train") or (args.mode == "Both"):
    def_meta_info = {"random_state": 1, "Training_Frac": 0.7, "Scaler": "Standard"}
    meta_info = deepcopy(def_meta_info)
    if args.meta_info is not None:
        for key, val in args.meta_info.items():
            if key in def_meta_info.keys():
                dtype = type(def_meta_info[key]) # get data type of value in default meta dict
                meta_info[key] = dtype(val)
            else:
                raise ValueError(f"{key} not found in default metadata dictionary, cannot determine data type of {val}")

if args.mode == "Test":
    resdir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir, "indiv_model_output")
    
    input_fname = os.path.join(resdir, f"{'_'.join([args.classifier, (args.sampling).replace(' ',''), args.optimizer])}_TrainedModelInfo.pkl")
    if not os.path.isfile(input_fname): raise ValueError(f"Could not find {input_fname}.")
    import pickle
    try:
        trained_results = pickle.load(open(input_fname, "rb"))
    except:
        raise ValueError(f"Could not open {input_fname}.")

    meta_info = mlflow.get_run(run_id=trained_results["Model_Info"][key].run_id).data.params

    pars_to_log = deepcopy(meta_info)
    pars_to_log.update({"Classifier" : args.classifier, "Sampling" : args.sampling, "Optimization" : args.optimizer})

features_train, features_test, target_train, target_test = an_steps.get_data(meta_info)


"""
Training Section
"""
if (args.mode == "Train") or (args.mode == "Both"):
    # get optimizer info
    if args.optimizer != "Default":
        import utils.model_optimization as optimize_model

    # define and make pipeline for resampling methods
    pipeline = model_opts.create_pipeline(args.sampling, **model_opts.get_resampling_method_pars(args.sampling, rand_state=meta_info["random_state"]))
    classifier = model_opts.create_classifier(args.classifier, **model_opts.get_classifier_pars(args.classifier, rand_state=meta_info["random_state"]))
    
    pars_to_log = deepcopy(classifier.get_params())
    pars_to_log.update(meta_info)
    pars_to_log.update({"Classifier" : args.classifier, "Sampling" : args.sampling, "Optimization" : args.optimizer})
    
    # perform resampling
        # for some reason this doesn't work when using only 1 line
    if pipeline is None:
        X_res, y_res = features_train.copy(), target_train.copy()
    else:
        X_res, y_res = pipeline.fit_resample(features_train.copy(), target_train.copy())
    
    print(f"\n\n\t\tTraining {key}\n")
    if args.optimizer == "Default":
        if args.no_results:
            trained_model_info = an_steps.models_logging("train", clone(classifier), X_res, y_res, pars_to_log, no_results=args.no_results)
            results_df = pd.DataFrame({key : mlflow.get_run(run_id=trained_model_info.run_id).data.metrics})
            print(f"\n\n---------- Model Training Completed ----------\n\n{results_df}")
        else:
            trained_model_info, results = an_steps.models_logging("train", clone(classifier), X_res, y_res, pars_to_log, no_results=args.no_results)
        
            from utils.compile_metrics import metrics2dict
        
            metrics_dict = metrics2dict(mlflow.get_run(run_id=trained_model_info.run_id).data.metrics)
            results.update(metrics_dict)
            
            results_df = pd.DataFrame({key : results})
            print(f"\n\n---------- Model Training Completed ----------\n\n{results_df}")
        
            outdict = {
                "Model_Info" : {key : trained_model_info},
                "Train_Results" : {
                    key : results,
                }
            }
        
            # save results as pickle file
            import pickle
            ## check if output directory exists and make it if it doesn't
            resdir = os.path.join(os.environ["RESULTS_DIR"], args.jobdir, "indiv_model_output")
            if not os.path.isdir(resdir): os.makedirs(resdir)
            outfname = os.path.join(resdir, f"{'_'.join([args.classifier, (args.sampling).replace(' ',''), args.optimizer])}_TrainedModelInfo.pkl")
        
            with open(outfname, "wb") as outfile:
                pickle.dump(outdict, outfile)
            print(f"{outfname} written")

    else:
        set_trace()
        clf = optimize_model.optimize_model(
            classifier=clone(classifier), algo=args.optimizer,
            data={"X_train" : X_res.copy(), "y_train" : y_res.copy(), "X_test" : features_test.copy(), "y_test" : target_test.copy()}
        )
        #    tmp_classifiers_dict[f"{key} {sampling_type} {optimizer_algo}"] = clf.best_estimator_


"""
Testing Section
"""
if (args.mode == "Test") or (args.mode == "Both"):
    print(f"\n\n\t\tTesting {key}\n")

    if args.no_results:
        tested_model_info = an_steps.models_logging("Test", trained_results["Model_Info"][key] if args.mode == "Test" else trained_model_info,
            features_test.copy(), target_test.copy(), pars_to_log, no_results=args.no_results
        )
        results_df = pd.DataFrame({key : mlflow.get_run(run_id=tested_model_info.run_id).data.metrics})
        print(f"\n\n---------- Model Testing Completed ----------\n\n{results_df}")
    
    else:
        tested_model_info, results = an_steps.models_logging("Test", trained_results["Model_Info"][key] if args.mode == "Test" else trained_model_info,
            features_test.copy(), target_test.copy(), pars_to_log, no_results=args.no_results
        )
    
        from utils.compile_metrics import metrics2dict
    
        metrics_dict = metrics2dict(mlflow.get_run(run_id=tested_model_info.run_id).data.metrics)
        results.update(metrics_dict)
    
        results_df = pd.DataFrame({key : results})
        print(f"\n\n---------- Model Testing Completed ----------\n\n{results_df}")
    
        outdict = {
            "Model_Info" : {key : tested_model_info},
            "Test_Results" : {
                key : results,
            }
        }
    
        ## save results as pickle file
        outfname = os.path.join(resdir, f"{'_'.join([args.classifier, (args.sampling).replace(' ',''), args.optimizer])}_TestedModelInfo.pkl")
        with open(outfname, "wb") as outfile:
            pickle.dump(outdict, outfile)
        print(f"{outfname} written")


toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
