from pdb import set_trace

from sklearn.base import clone # needed for 'initializing' models for each resampling method

def optimize_geneticsearchcv(classifier, X_train, y_train, param_grid=None, X_test=None, y_test=None):
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Categorical, Continuous
    def_param_grid = {
        "n_estimators": Integer(50, 150),
        "max_depth": Integer(5, 15),
        "learning_rate": Continuous(0.01, 1., distribution="log-uniform"),
        "gamma": Continuous(0.1, 100., distribution="log-uniform"),
        "reg_alpha": Continuous(0.1, 10., distribution="log-uniform"),
        "reg_lambda": Continuous(0.1, 10., distribution="log-uniform"),
    }

    def_model = clone(classifier)
        #GridSearch instance of current iteration
    clf = GASearchCV(estimator=def_model, param_grid=def_param_grid if param_grid is None else param_grid,
            scoring="f1", return_train_score=True, verbose=True, cv=5, n_jobs=-1, generations=10)
    clf.fit(X_train, y_train)

    return clf


def optimize_randomizedsearchcv(classifier, X_train, y_train, param_grid=None, X_test=None, y_test=None):
    from sklearn.model_selection import RandomizedSearchCV
    def_param_grid = {
        "learning_rate": [0.01, 0.1, 1.],
        "n_estimators": [50, 100, 150],
        "gamma": [0.1, 1., 10., 100.],
        "max_depth": [5, 10, 15],
        "reg_alpha": [0.1, 1., 10.],
        "reg_lambda": [0.1, 1., 10.],
    }

    def_model = clone(classifier)
        #GridSearch instance of current iteration
    clf = RandomizedSearchCV(estimator=def_model, param_distributions=def_param_grid if param_grid is None else param_grid,
            n_iter=100, scoring="f1", return_train_score=True, verbose=1, cv=5, n_jobs=-1)
    if (X_test is None) or (y_test is None):
        clf.fit(X_train, y_train, verbose=False)
    else:
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    return clf


def optimize_gridsearchcv(classifier, X_train, y_train, param_grid=None, X_test=None, y_test=None):
    from sklearn.model_selection import GridSearchCV
    def_param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 1.],
        "gamma": [0.1, 1., 10., 100.],
        "max_depth": [5, 10, 15],
        "reg_alpha": [0.1, 1., 10.],
        "reg_lambda": [0.1, 1., 10.],
    }

    def_model = clone(classifier)
        #GridSearch instance of current iteration
    clf = GridSearchCV(estimator=def_model, param_grid=def_param_grid if param_grid is None else param_grid,
            scoring="f1", return_train_score=True, verbose=1, cv=5, n_jobs=-1)
    if (X_test is None) or (y_test is None):
        clf.fit(X_train, y_train, verbose=False)
    else:
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    return clf



def optimize_model(classifier, algo:str, data=dict(X_train=None, y_train=None, X_test=None, y_test=None)):
    optim_algo_opts = {
        "GridSearchCV" : optimize_gridsearchcv,
        "RandomizedSearchCV" : optimize_randomizedsearchcv,
        "GASearchCV" : optimize_geneticsearchcv,
    }
    if algo not in optim_algo_opts.keys(): ValueError(f"Parameter optimization algorithm {algo} isn't currently supported! Must choose from {optim_algo_opts}")

    return optim_algo_opts[algo](classifier, **data)
