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
    - Evaluate the trained models created in step 4 using the unmodified (no resampling applied) testing data with the command `python src/model_testing.py jobdir [--debug] [--no_results] [--no_plots]` where `jobdir` is the same as the previous steps.
    - The results from testing the models are saved in `results/jobdir/TestingResults.pkl`.
    - Plots of the **Confusion Matrices**, **ROC** curves, **Precision-Recall** curves, and metrics using testing data are also created and saved in `results/jobdir/`.
6. Running steps 3-5 at once (optional):
    - Run command `bash prep_train_test.sh jobdir` in order to run preprocessing, model training, and model testing consecutively using the default config file.

## Discussion
Results running the command `bash prep_train_test.sh DefaultConfig` can be found in `results/DefaultConfig/`. The results are shown for the 16 different resampling+classifier model combinations, with the context that no parameter optimization was attempted for any of them.

### Model Training Results
```
---------- Model Training Completed ----------

                                       Cross_Val Precision    Recall        F1
LogisticRegression No Sampling          0.999197  0.896154  0.652661  0.755267
SGDClassifier No Sampling               0.999092  0.878049  0.605042  0.716418
DecisionTree No Sampling                0.999087       1.0       1.0       1.0
RandomForest No Sampling                0.999508       1.0       1.0       1.0
LogisticRegression Under-Sampling       0.935586  0.979351  0.929972  0.954023
SGDClassifier Under-Sampling            0.908914  0.917808  0.938375  0.927978
DecisionTree Under-Sampling             0.887964       1.0       1.0       1.0
RandomForest Under-Sampling             0.938373       1.0       1.0       1.0
LogisticRegression Over-Sampling        0.955002  0.980162  0.930028  0.954437
SGDClassifier Over-Sampling              0.95414  0.978972  0.932736  0.955295
DecisionTree Over-Sampling              0.999756       1.0       1.0       1.0
RandomForest Over-Sampling              0.999952       1.0       1.0       1.0
LogisticRegression Over+Under-Sampling  0.964824  0.979003  0.913769  0.945262
SGDClassifier Over+Under-Sampling       0.959045  0.984281  0.893618  0.936761
DecisionTree Over+Under-Sampling        0.998593       1.0       1.0       1.0
RandomForest Over+Under-Sampling        0.999832       1.0       1.0       1.0
```

As can be seen in the table and the training **Confusion Matrices**, the DecisionTree and RandomForest classifiers are able to correctly identify and classify the fraud ann normal transactions in the traiing set, regardless of resampling method. For the other two models, the **Precision**, **Recall**, and **F1-score** values all increase substantially when any type of resampling occurs.

### Model Testing Results
```
---------- Model Testing Completed ----------

                                       Precision    Recall        F1
LogisticRegression No Sampling          0.840426  0.585185  0.689956
SGDClassifier No Sampling               0.826087  0.562963  0.669604
DecisionTree No Sampling                0.733333  0.733333  0.733333
RandomForest No Sampling                0.890756  0.785185  0.834646
LogisticRegression Under-Sampling       0.042732  0.903704  0.081605
SGDClassifier Under-Sampling            0.017888  0.911111  0.035088
DecisionTree Under-Sampling             0.013958  0.888889  0.027485
RandomForest Under-Sampling             0.056505  0.881481  0.106203
LogisticRegression Over-Sampling        0.070885  0.896296  0.131379
SGDClassifier Over-Sampling              0.06612  0.896296  0.123155
DecisionTree Over-Sampling              0.741935  0.681481  0.710425
RandomForest Over-Sampling              0.929825  0.785185  0.851406
LogisticRegression Over+Under-Sampling  0.126087  0.859259  0.219905
SGDClassifier Over+Under-Sampling        0.16259  0.837037  0.272289
DecisionTree Over+Under-Sampling        0.449339  0.755556  0.563536
RandomForest Over+Under-Sampling        0.797101  0.814815  0.805861
```

From this table, it can be seen that the RandomForest classifier has the highest **F1-score** for a given resampling method, and usually by a significant margin. Whereas during the training the "No Sampling" scores were significantly less than the other resampling techniques, the **F1-score** values during testing were much smaller for "Under-Sampling" than the others.

Context is extremely important when analyzing the **ROC** curves. Real-world fraud detection systems usually handle hundreds of thousands to millions of on a daily basis, and so any False Positive Rate higher than 0.1% is too high and most of the **ROC** curve phase space is irrelevant. This is why a "zoomed in" plot of the **ROC** curves below this FPR was created. Seen in this plot, the "RandomForest Over+Under-Sampling" model has the best performance of all. 

Turning our attention to the **Precision-Recall** curves, we can see that the RandomForest classifiers perform the best for each sampling technique, with the highest average precision (AP) score belonging to the "RandomForest Over-Sampling" model.

Overall, I would say the best-performing model is the "RandomForest Over+Under-Sampling", with the caveat that no parameter optimization was performed for any of the models. Perhaps in the future this will be done, or other classifiers will be supported with which to compare results.
