from pdb import set_trace

def csv_to_pandas_DF(fname, **opts):
    """
    Read the data file if it exists and return it as a pandas DataFrame
    """
    import pandas as pd
    import os

    fpath = os.path.join("data", fname)
    try:
        pd_df = pd.read_csv(fpath, **opts)
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

    if model_type == "DecisionTree":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(**model_opts)

    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**model_opts)

    # Fit model
    model.fit(train_X, train_y)

    return model


def make_lags(ts, lags, lead_time=1):
    import pandas as pd

    return pd.concat(
        {
            f"y_lag_{i}": ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)

def make_multistep_target(ts, steps, reverse=True):
    import pandas as pd

    range_it = reversed(range(steps)) if reverse else range(steps)
    return pd.concat(
        {f"y_step_{i + 1}": ts.shift(-i)
         for i in range_it},
        axis=1)

