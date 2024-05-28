from pdb import set_trace

def pipeline_dict_constructor(**opts):
    """
    This function returns a dictionary of pipeline options for different resampling methods.
    """
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler

    pipeline_options = {}
    if "No Sampling" in opts.keys():
        pipeline_options["No Sampling"] = None
    if "Under-Sampling" in opts.keys():
        pipeline_options["Under-Sampling"] = Pipeline(steps=[
            ("under", RandomUnderSampler(**opts["Under-Sampling"])),
        ])
    if "Over-Sampling" in opts.keys():
        pipeline_options["Over-Sampling"] = Pipeline(steps=[
            ("over", RandomOverSampler(**opts["Over-Sampling"])),
        ])
    if "Over+Under-Sampling" in opts.keys():
        pipeline_options["Over+Under-Sampling"] = Pipeline(steps=[
            ("over", RandomOverSampler(**opts["Over+Under-Sampling"]["over"])),
            ("under", RandomUnderSampler(**opts["Over+Under-Sampling"]["under"])),
        ])

    return pipeline_options


def classifiers_dict_constructor(**opts):
    classifiers = {}
    if "LogisticRegression" in opts.keys():
        from sklearn.linear_model import LogisticRegression
        classifiers["LogisticRegression"] = LogisticRegression(**opts["LogisticRegression"])

    if "SGDClassifier" in opts.keys():
        from sklearn.linear_model import SGDClassifier
        classifiers["SGDClassifier"] = SGDClassifier(**opts["SGDClassifier"])

    if "DecisionTree" in opts.keys():
        from sklearn.tree import DecisionTreeClassifier
        classifiers["DecisionTree"] = DecisionTreeClassifier(**opts["DecisionTree"])

    if "RandomForest" in opts.keys():
        from sklearn.ensemble import RandomForestClassifier
        classifiers["RandomForest"] = RandomForestClassifier(**opts["RandomForest"])

    if "KNearest" in opts.keys():
        raise ValueError("This classifier option isn't currently supported!")
        #from sklearn.neighbors import KNeighborsClassifier
        #classifiers["KNearest"] = KNeighborsClassifier(**opts["KNearest"])

    if "LinearSVC" in opts.keys():
        raise ValueError("This classifier option isn't currently supported!")
        #from sklearn.svm import LinearSVC
        #classifiers["LinearSVC"] = LinearSVC(**opts["LinearSVC"])

    if "SVC" in opts.keys():
        raise ValueError("This classifier option isn't currently supported!")
        #from sklearn.svm import SVC
        #classifiers["SVC"] = SVC(**opts["SVC"])
        ###"SVC": SVC(random_state=rand_state, probability=True, kernel="linear"), # set prob to True so 'predict_proba' can be used

    if ("XGBoost" in opts.keys()) or ("XGB" in opts.keys()):
        from xgboost import XGBClassifier
        classifiers["XGB"] = XGBClassifier(**opts["XGB"])

    return classifiers
