from pdb import set_trace

def csv_to_pandas_DF(fname):
    """
    Read the data file if it exists and return it as a pandas DataFrame
    """
    import pandas as pd
    import os

    fpath = os.path.join("data", fname)
    try:
        pd_df = pd.read_csv(fpath)
    except:
        raise ValueError(f"{fpath} could not be found")

    return pd_df



def get_mae(model, val_X, val_y):
    """
    Calculates Mean Absolute Error between y values from model prediction and validation set.
    """
    from sklearn.metrics import mean_absolute_error

    # get predicted values on validation data
    pred_vals = model.predict(val_X)
    # calculate the mean absolute error
    mae = mean_absolute_error(val_y, pred_vals)

    return mae



def fit_ml_model(model_type, train_X, train_y, **model_opts):
    model_choices_ = ["DecisionTree", "RandomForest"]
    if model_type not in model_choices_:
        raise ValueError(f"Specified model_type {model_type} is not in {model_choices_}.")

    rand_state = model_opts.get("random_state", 0)

    if model_type == "DecisionTree":
        from sklearn.tree import DecisionTreeRegressor
        max_leafs = model_opts.get("max_leaf_nodes", None)
        model = DecisionTreeRegressor(max_leaf_nodes = max_leafs, random_state = rand_state)

    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state = rand_state)

    # Fit model
    model.fit(train_X, train_y)

    return model
