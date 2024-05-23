## Introduction
This project uses various machine learning models to classify fraudulent credit card transactions given input data.


## Data
This project utilizes the "Credit Card Fraud Detection" [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) which has already been preprocessed. As stated on the kaggle page, features **V1, V2, ... V28** have been obtained using PCA, with **Time** and **Amount** as (untransformed) additional features. As can be seen from exploring the dataset, it is highly unbalanced with only 0.17% of all transactions being fraudulent.


## Approach
1. The **Time** and **Amount** features will be transformed using either the **StandardScaler** or **RobustScaler** techniques, and then split the features and target into training and test sets.
2. Since the data is extremely unbalanced, different sampling methods will be employed. The supported options are "No Sampling", "Under-Sampling", "Over-Sampling", and "Over+Under-Sampling". Resampling is done using the **RandomUnderSampler** and **RandomOverSampler** methods from `imblearn`. Although there are many other options to use individual sampling methods or combinations of them, the default methods are sufficient for these studies.
3. The effectiveness of each sampling method are evaluated using different classifiers. Currently, the four supported classifiers are:
    - LogisticRegression
    - SGDClassifier
    - DecisionTree
    - RandomForest
4. The metrics which will be used to evaluate the different sampling-classifier combination are **Precision**, **Recall**, and **F1-score**. **Precision** is a model evaluation and performance metric that corresponds to the fraction of values that actually belong to a positive class out of all of the values which are predicted to belong to that class.
    `Precision = Number of True Positives / (Number of True Positives + False Positives) = TP / (TP + FP)`
    **Recall** corresponds to the fraction of values predicted to be of a positive class out of all the values that truly belong to the positive class (including false negatives). 
    `Recall = Number of True Positives / (Number of True Positives + False Negatives) = TP / (TP + FN)`
     **F1-score** is a useful metric for measuring the performance for classification models using imbalanced data because it takes into account the type of errors, false positive and false negative, and not just the number of predictions that were incorrect, a necessity in areas like fraud prevention and other industry use cases. **F1-score** computes the harmonic average of precision and recall, where the relative contribution of both of these metrics are equal to **F1-score**.
As a result,
    `F1  = 2 * TP / (2 * TP + FP + FN)`
    During model training, the **cross_val_score** using 5-fold cross validation is also computed. During testing, the **Receiver Operating Characteristic (ROC)** and **Precision-Recall** curves for each model are calculated, as well as the area under the curves.
    **Confusion Matrices** for binary classification are also created during training and testing, which are a graphical representation of how models classify data as True Positives, True Negatives, False Positives, and False Negatives.


## Running the Code
1. Get the data:
    - Download the "Credit Card Fraud Detection" dataset discussed above and save it in the `data/` directory.
2. Set global variables:
    - Run the `source environment.sh` command to set environmental variables used throughout the project.
3. Preprocess the data:
    - Preprocess the data and save the output running the command `python src/preprocessing.py jobdir [--cfile CFILE] [--debug] [--no_results] [--no_plots]` where `jobdir` is the name of the output directory created in `results/`.
    - `CFILE` is the name of the config file which specifies various values (such as the random state and training/testing fraction) for reproducibitily, as well as which sampling techniques and classifiers to train and test. `configs/default_config.json` is the default file to use unless another config file is created and specified.
4. Train the models:
    - Using the preprocessed data file from the previous step, run `python src/model_training.py jobdir [--debug] [--no_results] [--no_plots]` where `jobdir` is the same as in step 3.
    - This script trains the resampling+classifier model combinations, specified in the config file, on the training data.
    - The trained models and the results of different metrics are saved in the `results/jobdir/TrainingResults.pkl` file.
    - Plots of the confusion matrices and training metrics for each model are also created and saved in `results/jobdir/`.
5. Test the models:
    - Evaluate the trained models created in step 4 using the testing data with the command `python src/model_testing.py jobdir [--debug] [--no_results] [--no_plots]` where `jobdir` is the same as the previous steps.
    - The results from testing the models are saved in `results/jobdir/TestingResults.pkl`.
    - Plots of the **Confusion Matrices**, **ROC** curves, **Precision-Recall** curves, and metrics using testing data are also created and saved in `results/jobdir/`.
6. Running steps 3-5 at once (optional):
    - Run command `bash prep_train_test.sh jobdir` in order to run preprocessing, model training, and model testing consecutively using the default config file.

## Discussion
