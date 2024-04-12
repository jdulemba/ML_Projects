import time
tic = time.time()

"""
Data leakage (or leakage) happens when your training data contains information about the target,
but similar data will not be available when the model is used for prediction. 
This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.

In other words, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.

There are two main types of leakage: target leakage and train-test contamination.

Target Leakage:
    Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
    It is important to think about target leakage in terms of the timing or chronological order that data becomes available,
    not merely whether a feature helps make good predictions.

    To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.


Train-Test Contamination:
    A different type of leak occurs when you aren't careful to distinguish training data from validation data.

    Recall that validation is meant to be a measure of how the model does on data that it hasn't considered before. 
    You can corrupt this process in subtle ways if the validation data affects the preprocessing behavior.
    This is sometimes called train-test contamination.
    
    For example, imagine you run preprocessing (like fitting an imputer for missing values) before calling train_test_split(). The end result?
    Your model may get good validation scores, giving you great confidence in it, but perform poorly when you deploy it to make decisions.
    
    After all, you incorporated data from the validation or test data into how you make predictions,
    so it may do well on that particular data even if it can't generalize to new data.
    This problem becomes even more subtle (and more dangerous) when you do more complex feature engineering.
    
    If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, including the fitting of preprocessing steps.
    This is easier if you use scikit-learn pipelines. When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline!
"""

import utils.Utilities as Utils
from pdb import set_trace
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Read the data
data = Utils.csv_to_pandas_DF("AER_credit_card_data.csv", **{"true_values" : ["yes"], "false_values" : ["no"]})

# Select target
y = data.card

# Select predictors
X = data.drop(["card"], axis=1)

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators = 100, random_state = 0))
cv_scores = cross_val_score(my_pipeline, X, y, cv = 5, scoring = "accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean(): .5f}")

"""
Cross-validation accuracy: 0.98029

With experience, you'll find that it's very rare to find models that are accurate 98% of the time.
It happens, but it's uncommon enough that we should inspect the data more closely for target leakage.
"""

# look at one of the variables
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print("Fraction of those who did not receive a card and had no expenditures: %.2f" \
      %((expenditures_noncardholders == 0).mean()))
print("Fraction of those who received a card and had no expenditures: %.2f" \
      %(( expenditures_cardholders == 0).mean()))

"""
The results of the above print statements are

Fraction of those who did not receive a card and had no expenditures: 1.00
Fraction of those who received a card and had no expenditures: 0.02

As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures.
It's not surprising that our model appeared to have a high accuracy.
But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

Since 'share' is partially determined by 'expenditure', it should be excluded too.
The variables 'active' and' majorcards' are a little less clear, but from the description, they sound concerning.
In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.
"""

## rerun the model without the target leakage
potential_leaks = ["expenditure", "share", "active", "majorcards"]
X2 = X.drop(potential_leaks, axis = 1)

# evaluate the model with the leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, cv = 5, scoring = "accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean(): .5f}")

"""
Cross-validation accuracy: 0.82941

This accuracy is quite a bit lower, which might be disappointing.
However, we can expect it to be right about 80% of the time when used on new applications,
whereas the leaky model would likely do much worse than that (in spite of its higher apparent score in cross-validation).
"""

toc = time.time()
print("Total runtime: %.2f" % (toc - tic))
