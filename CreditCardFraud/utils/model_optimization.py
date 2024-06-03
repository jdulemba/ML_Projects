from pdb import set_trace

from sklearn.base import clone # needed for 'initializing' models for each resampling method

#setting grid of selected parameters for iteration
xgb_param_grid = {
    "gamma": [0.1, 1., 10., 100.],
    "learning_rate": [0.01, 0.1, 1.],
    "max_depth": [5, 10, 15],
    "n_estimators": [50, 100, 150],
    "reg_alpha": [0.1, 1., 10.],
    "reg_lambda": [0.1, 1., 10.],
}


def optimize_gridsearchcv(classifier, X_train, y_train, X_test=None, y_test=None):
    from sklearn.model_selection import GridSearchCV
    def_model = clone(classifier)
        #GridSearch instance of current iteration
    clf = GridSearchCV(estimator=def_model, param_grid=xgb_param_grid, scoring="f1", return_train_score=True, verbose=1, cv=5, n_jobs=-1)
    if (X_test is None) or (y_test is None):
        clf.fit(X_train, y_train, verbose=False)
    else:
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    return clf



def optimize_model(classifier, algo:str, data=dict(X_train=None, y_train=None, X_test=None, y_test=None)):
    optim_algo_opts = {
        "GridSearchCV" : optimize_gridsearchcv,
    }
    if algo not in optim_algo_opts.keys(): ValueError(f"Parameter optimization algorithm {algo} isn't currently supported! Must choose from {optim_algo_opts}")

    return optim_algo_opts[algo](classifier, **data)
