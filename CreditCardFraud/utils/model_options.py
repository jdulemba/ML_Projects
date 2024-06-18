from pdb import set_trace


resampling_opts_ = ["No Sampling", "Under-Sampling", "Over-Sampling", "Over+Under-Sampling"]
scaler_opts_ = ["Standard", "Robust"]
classifier_opts_ = ["LogisticRegression", "SGDClassifier", "DecisionTree", "RandomForest", "XGBoost"]


# add random state value into options dict
def add_random_state_to_dict(input_dict, rand_state):
    """This function adds the 'random_state' key and value pair to the deepest levels of the input dictionary"""
    if input_dict == dict():
        input_dict["random_state"] = rand_state
    else:
        for k, v in input_dict.copy().items():
            if isinstance(v, dict):     # For DICT
                if not v: # if dict is empty
                    v["random_state"] = rand_state
                else:
                    input_dict[k] = add_random_state_to_dict(v, rand_state)
            else: # Update Key-Value
                input_dict["random_state"] = rand_state

    return input_dict


def get_scaler(scaler_type):
    assert scaler_type in scaler_opts_, f"Scaler_type must be one of {scaler_opts_}"
    if scaler_type == "Standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_type == "Robust":
        ## RobustScaler is less prone to outliers.
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise IOError(f"Scaler type {scaler_type} is not a supported option.\nOnly {scaler_opts_} are currently available.")

    return scaler


def get_resampling_method_pars(method, rand_state=None):
    """
    This function returns a dictionary of default options for a given resampling method.
    """
    assert method in resampling_opts_, f"Resampling method must be one of {resampling_opts_}"
    default_resampling_pars = {
        "No Sampling" : {},
        "Under-Sampling" : {"sampling_strategy" : "majority"},
        "Over-Sampling" : {"sampling_strategy" : "minority"},
        "Over+Under-Sampling" : {
            "over" : {"sampling_strategy" : 0.1},
            "under" : {"sampling_strategy" : 0.5}
        }
    }

    resampling_opts = default_resampling_pars[method]
    if rand_state: resampling_opts = add_random_state_to_dict(resampling_opts, rand_state)

    return resampling_opts


def create_pipeline(method, **opts):
    """
    This function returns a dictionary of pipeline options for different resampling methods.
    """
    assert method in resampling_opts_, f"Resampling method must be one of {resampling_opts_}"

    if not opts:
        opts = get_resampling_method_pars(method)

    if method == "No Sampling":
        pipeline = None
    elif method == "Under-Sampling":
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
        pipeline = Pipeline(steps=[
            ("under", RandomUnderSampler(**opts)),
        ])
    elif method == "Over-Sampling":
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import RandomOverSampler
        pipeline = Pipeline(steps=[
            ("over", RandomOverSampler(**opts)),
        ])
    elif method == "Over+Under-Sampling":
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import RandomOverSampler
        pipeline = Pipeline(steps=[
            ("over", RandomOverSampler(**opts["over"])),
            ("under", RandomUnderSampler(**opts["under"])),
        ])

    return pipeline


def get_classifier_pars(clf_type, rand_state=None):
    assert clf_type in classifier_opts_, f"{clf_type} is not supported, must choose one of {classifier_opts_}."

    default_class_opts = {
        "LogisticRegression" : {},
        "SGDClassifier" : {"loss" : "log_loss"},
        "DecisionTree" : {},
        "RandomForest" : {},
        "XGBoost" : {}
    }
    classifier_opts = default_class_opts[clf_type]
    if rand_state: classifier_opts = add_random_state_to_dict(classifier_opts, rand_state)

    return classifier_opts


def create_classifier(clf_type, **opts):
    assert clf_type in classifier_opts_, f"{clf_type} is not supported, must choose one of {classifier_opts_}."

    if not opts:
        opts = get_classifier_pars(clf_type)

    if clf_type == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**opts)

    elif clf_type == "SGDClassifier":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(**opts)

    elif clf_type == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**opts)

    elif clf_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**opts)

    elif clf_type == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**opts)

    #return classifiers


