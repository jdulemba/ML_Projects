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


## Running the Code

