from pdb import set_trace
import numpy as np
import pandas as pd

def get_data(meta_info):
        # get variables from meta info
    rand_state = int(meta_info["random_state"]) if "random_state" in meta_info.keys() else None
    scaler_type = meta_info.get("Scaler")
    if "Training_Frac" in meta_info.keys():
        training_frac = float(meta_info["Training_Frac"])
        testing_frac = 1 - training_frac
    else:
        training_frac, testing_frac = None, None

    # set random seeds for reproducibility
    np.random.seed(rand_state)
    
    # open data
    import os
    data = pd.read_csv(os.path.join(os.environ["PROJECT_DIR"], "data", "creditcard.csv"))
    # check if there are any null values
    assert not data.isnull().any().any(), "There are unexpected Null values in the dataset"
    
    # determine how many cases are fraud/not fraud
    target_name = "Class"
    
    ## Scale 'Time' and 'Amount' features, and compare distributions before/after scaling
    import utils.model_options as model_opts
    scaler = model_opts.get_scaler(scaler_type=scaler_type)
    
    data["scaled_amount"] = scaler.fit_transform(data["Amount"].values.reshape(-1,1))
    data["scaled_time"] = scaler.fit_transform(data["Time"].values.reshape(-1,1))
    
    # drop original features which were scaled
    data = data.drop(["Time","Amount"], axis=1)
    
    # determine features and target for entire dataset
    features, target = data.iloc[:, data.columns != target_name], data[target_name]
    
    # split into training and testing sets
    from sklearn.model_selection import train_test_split
    features_train, features_test, target_train, target_test = train_test_split(features, target,
        train_size=training_frac, test_size=testing_frac, random_state=rand_state,
    )

    return features_train, features_test, target_train, target_test


# Create a new MLflow Experiment
import mlflow
mlflow_artifact_path = "credit_card_fraud"

def models_logging(data_type, input_clf, X, y, model_params, no_results=False, optimize=False):
    from sklearn.metrics import roc_curve, precision_recall_curve
    from mlflow.models import infer_signature

    assert data_type.capitalize() in ["Train", "Test"], f"{data_type.capitalize()} is not a valid type, must choose from ['Train', 'Test']"
    if data_type.capitalize() == "Train":
        from sklearn.base import clone
        classifier = clone(input_clf)
        classifier.fit(X.copy(), y.copy())
        from sklearn.model_selection import cross_val_score
        cross_val_scores = cross_val_score(classifier, X.copy(), y.copy(), cv=5)
        
    if data_type.capitalize() == "Test":
        classifier = mlflow.sklearn.load_model(input_clf.model_uri)

    key = f"{model_params['Classifier']} {model_params['Sampling']} {model_params['Optimization']}"

    precision_array, recall_array, pr_thresh = precision_recall_curve(y, classifier.predict_proba(X)[:, 1])
    fpr_array, tpr_array, roc_thresh = roc_curve(y, classifier.predict_proba(X)[:, 1])

    if not no_results:
        curve_results = {
            "ROC_FPR" : fpr_array, "ROC_TPR" : tpr_array, "ROC_Thresh" : roc_thresh,
            "PRCurve_Precision" : precision_array, "PRCurve_Recall" : recall_array, "PRCurve_Thresh" : pr_thresh,
        }

    # Infer the model signature
    signature = infer_signature(X.copy(), classifier.predict(X.copy()))

    # Build the Evaluation Dataset from the test set
    eval_data = X.copy()
    eval_data["label"] = y.copy()

    # Start an MLflow run
    with mlflow.start_run(nested=False):
        # Log the hyperparameters
        mlflow.log_params(model_params)

        if data_type.capitalize() == "Train":
            # Log the cross-validation metric
            mlflow.log_metric("Cross_Val", cross_val_scores.mean())

        # Log the precision-recall curve info
        mlflow.log_dict(pd.DataFrame({"PRCurve_Precision" : precision_array, "PRCurve_Recall" : recall_array}).to_dict(orient="dict"), "PRCurve_results.json")
        # Log the ROC curve info
        mlflow.log_dict(pd.DataFrame({"ROC_FPR" : fpr_array, "ROC_TPR" : tpr_array}).to_dict(orient="dict"), "ROCCurve_results.json")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(f"{data_type.capitalize()}ing Info", key)

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=classifier,
            artifact_path=mlflow_artifact_path,
            signature=signature,
            input_example=X.copy(),
            registered_model_name=f"{data_type.capitalize()}ing: {key}",
        )

        # evaluate the logged model
        model_uri = mlflow.get_artifact_uri(mlflow_artifact_path)
        mlflow.evaluate(
            model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators="default",
            evaluator_config={"log_model_explainability": False if optimize else not ((model_params["Classifier"] == "DecisionTree") or (model_params["Classifier"] == "RandomForest"))},

        )

    if not no_results:
        return model_info, curve_results
    else:
        return model_info
