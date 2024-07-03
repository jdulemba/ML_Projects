## Introduction
This project uses various machine learning models to classify fraudulent credit card transactions given input data. Individual model logging and performance are tracked using [mlflow](https://mlflow.org).


## Data
This project utilizes the “Credit Card Fraud Detection” [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which has already been preprocessed. As stated on the Kaggle page, features **V1, V2, ... V28** have been obtained using PCA, with **Time** and **Amount** as (untransformed) additional features. As can be seen from exploring the dataset, it is highly unbalanced, with only 0.17% of all transactions being fraudulent.


## Approach
1. The **Time** and **Amount** features will be transformed using either the **StandardScaler** (default) or **RobustScaler** techniques, and then split into training and test sets.

2. Since the data is extremely unbalanced, different resampling methods will be employed. The supported options are “No Sampling,” “Under-Sampling,” “Over-Sampling,” and “Over+Under-Sampling.” Resampling is done using the **RandomUnderSampler** and **RandomOverSampler** methods from `imblearn`. Although there are many other options to use individual sampling methods or combinations of them, the default methods are sufficient for these studies.

3. The effectiveness of each resampling method is evaluated using different five different classifiers.
    - DecisionTreeClassifier
    - LogisticRegression
    - RandomForestClassifier
    - SGDClassifier
    - XGBClassifier

4. The metrics used to evaluate the different resampling-classifier combinations are the **F1** score, the average precision from the **Precision-Recall** curve (**PR AP**), and the area under the **Receiver Operating Characteristic** curve (**ROC AUC**). **Precision** is a model evaluation and performance metric that corresponds to the fraction of values that actually belong to the positive class out of all of the values predicted to belong to that class.

    `Precision = Number of True Positives / (Number of True Positives + False Positives) = TP / (TP + FP)`.

    **Recall** corresponds to the fraction of values predicted to be of the positive class out of all the values that truly belong to the positive class (including false negatives).

    `Recall = Number of True Positives / (Number of True Positives + False Negatives) = TP / (TP + FN)`.

    **F1** score is a useful metric for measuring the performance of classification models using imbalanced data because it takes into account the type of errors, false positives and false negatives, and not just the number of predictions that were incorrect, a necessity in areas like fraud prevention and other industry use cases. **F1** score computes the harmonic average of **Precision** and **Recall**, where the relative contributions of both metrics are equal to F1-score. As a result,

    `F1 = 2 * TP / (2 * TP + FP + FN)`.

    Context is extremely important when analyzing the **ROC** curves. Real-world fraud detection systems usually handle hundreds of thousands to millions of transactions daily, and so any False Positive Rate (FPR) higher than 0.1% is too high, making most of the **ROC** curve phase space irrelevant. With this in mind, the **ROC AUC** with FPR < 0.001 is used as the evaluating metric instead of the area from the entire phase space.

    During model training and testing, the **ROC** and **Precision-Recall** curves for each model are calculated, as well as the area under the curves. During model training, the **cross_val_score** using 5-fold cross-validation is also computed.

    **Confusion Matrices** for binary classification are also created during training and testing, which are graphical representations of how models classify data as True Positives, True Negatives, False Positives, and False Negatives.

5. Rank each of the 20 models based on their testing performance in the **F1**, **PR AP**, and **ROC AUC < 0.001** metrics and choose the best model by using a weighted average of these rankings. Since this analysis does not take into account any specific cost or risk factors that come with misidentifying fraudulent and normal transactions, the three metrics are equally weighted.

## Running the Code
1. Get the data.
    - Download the "Credit Card Fraud Detection" dataset discussed above and save it in the `data/` directory.

2. Set global variables.
    - Run the command `source environment.sh` to set environmental variables used throughout the project.

3. Start a local MLflow server using the same value as the `MLFLOW_TRACKING_URI` variable, which will be used for model logging.
    - Run the command `mlflow server --host 127.0.0.1 --port 8080`.

4. The ability to process the data, train a model, and test a model is contained within the `src/mlflow_train_test.py` file.
    - In order to do all of the steps at once, run the command `python src/mlflow_train_test.py jobdir classifier sampling optimizer [--mode {Train,Test,Both}] [--meta_info [META_INFO ...]] [--no_results]`.
    - `jobdir` is the name of the output directory created in `results/`.
    - `classifier` specifies the type of classication model to be trained, chosen from `LogisticRegression, SGDClassifier, DecisionTree, RandomForest, XGBoost`.
    - `sampling` specifies the resampling technique used, chosen from `No Sampling, Under-Sampling, Over-Sampling, Over+Under-Sampling`.
    - `optimizer` specifies which hyperparameter optimization algorithm to use, chosen from `Default, GridSearchCV, RandomizedSearchCV, GASearchCV`.
    - `mode` specifies whether to run only model training (`Train`), model testing (`Test`), or both consecutively (`Both`). This parameter is defaulted to `Both`.
    - `meta_info` allows for parameters relevant to the analysis (random state, training/testing fraction, etc.) to be specified from the command line through key-value pairs separated by '=', i.e. `--meta_info random_state=22 Training_Frac=0.9`.
    - `no_results` determines whether making an output pkl file containing the results is suprressed or not (default is `False`). MLflow logging will still occur even if this is `True`.

    Running this script for a single classifier-resampling model combination results in a corresponding file containing a dictionary of values to be plotted being saved in `results/jobdir/indiv_model_output/`.

5. Combine all individual model files into a single file which has the results from training and testing.
    - Run the command `python utils/combine_results.py jobdir res_type`, where `res_type` specifies whether to combine all model training (`Train`) or testing (`Test`) files.

6. Plot the results running the command `src/plot_results.py jobdir res_type [--grouping]`.
    - `res_type` specifies whether to plot results from model training (`Train`) or testing (`Test`).
    - `grouping` specifies how results should be grouped for plotting (by Classifier, Sampling, or Optimization type).

    Plots of **Confusion Matrices**, **Precision-Recall** curves, **ROC** curves, and other metrics (**F1**, **precision**, **recall**, **cross-validation**) are output in the `results/jobdir/Training/` or `results/jobdir/Testing/` directories, with subdirectories corresponding to the specified grouping.

7. Rank the models based on their testing performance across several different metrics, equally weighing the **F1**, **PR AP**, and **ROC AUC < 0.001** values.
    - Run the command `python src/rank_models.py jobdir`.

8. Run steps 4-7 in one command (optional):
    - Train and test all of the models using default hyperparameters, plot the results, and rank the models with the command `bash run_entire_def_analysis.sh jobdir`, which uses the default analysis parameters and model hyperparameters.
    - This command takes approximately 30 minutes to run.


## Discussion
Results running the command `bash run_entire_def_analysis.sh MLflowDefaultSetup` from step 8 can be found in `results/MLflowDefaultSetup/`. The results are shown for the 20 different model combinations using the default hyperparameters for each classifier.

### Model Training Results
```
                                       Cross_Val   Precision   Recall      F1
DecisionTree No Sampling                0.999087    1.0         1.0         1.0
DecisionTree Over+Under-Sampling        0.998593    1.0         1.0         1.0
DecisionTree Over-Sampling              0.999756    1.0         1.0         1.0
DecisionTree Under-Sampling             0.887964    1.0         1.0         1.0
LogisticRegression No Sampling          0.999197    0.896154    0.652661    0.755267
LogisticRegression Over+Under-Sampling  0.964824    0.979003    0.913769    0.945262
LogisticRegression Over-Sampling        0.955002    0.980162    0.930028    0.954437
LogisticRegression Under-Sampling       0.935586    0.979351    0.929972    0.954023
RandomForest No Sampling                0.999508    1.0         1.0         1.0
RandomForest Over+Under-Sampling        0.999832    1.0         1.0         1.0
RandomForest Over-Sampling              0.999952    1.0         1.0         1.0
RandomForest Under-Sampling             0.938373    1.0         1.0         1.0
SGDClassifier No Sampling               0.999092    0.878049    0.605042    0.716418
SGDClassifier Over+Under-Sampling       0.959045    0.984281    0.893618    0.936761
SGDClassifier Over-Sampling             0.95414     0.978972    0.932736    0.955295
SGDClassifier Under-Sampling            0.908914    0.917808    0.938375    0.927978
XGBoost No Sampling                     0.999534    1.0         1.0         1.0
XGBoost Over+Under-Sampling             0.999631    1.0         1.0         1.0
XGBoost Over-Sampling                   0.99993     1.0         1.0         1.0
XGBoost Under-Sampling                  0.95098     1.0         1.0         1.0
```

The table above shows the cross-validation, **Precision**, **Recall** and **F1** values resulting from model training, allowing for several conclusions to be drawn.

1. DecisionTree, RandomForest, and XGBoost Classifiers:
    - Achieved perfect classification metrics (**Precision**, **Recall**, **F1**) for all of the resampling methods, demonstrating their effectiveness for this task.
    - These classifiers also maintained cross-validation scores around 0.999 or higher for most resampling methods, indicating robust model performance during training.

2. LogisticRegression and SGDClassifier Classifiers:
    - Showed strong performance with resampling methods, emphasizing the importance of resampling in managing imbalanced datasets for these classifiers.
    - Lower performance without sampling, particularly in **Recall** and **F1** metrics, suggests these models benefit significantly from resampling.

3. Under-Sampling:
    - Cross-validation scores with Under-Sampling were significantly lower, indicating potential overfitting or less stability during training compared to other resampling methods.
    - This may hint that the limited statistics resulting from this method might not be able to generalize to the testing dataset.

4. Consistent High Performance:
    - XGBoost consistently performed exceptionally well across all resampling methods, making it a reliable choice for this classification task.

### Model Testing Results
```
---------- Values ----------
                                       Precision   Recall      F1
DecisionTree No Sampling                0.733333    0.733333    0.733333
DecisionTree Over+Under-Sampling        0.449339    0.755556    0.563536
DecisionTree Over-Sampling              0.741935    0.681481    0.710425
DecisionTree Under-Sampling             0.013958    0.888889    0.027485
LogisticRegression No Sampling          0.840426    0.585185    0.689956
LogisticRegression Over+Under-Sampling  0.126087    0.859259    0.219905
LogisticRegression Over-Sampling        0.070885    0.896296    0.131379
LogisticRegression Under-Sampling       0.042732    0.903704    0.081605
RandomForest No Sampling                0.890756    0.785185    0.834646
RandomForest Over+Under-Sampling        0.797101    0.814815    0.805861
RandomForest Over-Sampling              0.929825    0.785185    0.851406
RandomForest Under-Sampling             0.056505    0.881481    0.106203
SGDClassifier No Sampling               0.826087    0.562963    0.669604
SGDClassifier Over+Under-Sampling       0.16259     0.837037    0.272289
SGDClassifier Over-Sampling             0.06612     0.896296    0.123155
SGDClassifier Under-Sampling            0.017888    0.911111    0.035088
XGBoost No Sampling                     0.904348    0.77037     0.832
XGBoost Over+Under-Sampling             0.705128    0.814815    0.756014
XGBoost Over-Sampling                   0.892562    0.8         0.84375
XGBoost Under-Sampling                  0.038826    0.911111    0.074478

---------- Rankings ----------
                                        F1 Rank  PR AP Rank  ROC AUC < 0.001 Rank  Average Rank
XGBoost Over-Sampling                      2         2               1                1.666667
XGBoost No Sampling                        4         1               2                2.333333
RandomForest Over-Sampling                 1         3               3                2.333333
RandomForest No Sampling                   3         5               5                4.333333
XGBoost Over+Under-Sampling                6         4               4                4.666667
RandomForest Over+Under-Sampling           5         6               6                5.666667
LogisticRegression No Sampling             9         7               7                7.666667
LogisticRegression Over+Under-Sampling    13         8               8                9.666667
SGDClassifier Over+Under-Sampling         12        12               9               11.000000
SGDClassifier No Sampling                 10        11              13               11.333333
SGDClassifier Over-Sampling               15         9              10               11.333333
LogisticRegression Over-Sampling          14        10              12               12.000000
DecisionTree No Sampling                   7        15              15               12.333333
DecisionTree Over-Sampling                 8        16              16               13.333333
RandomForest Under-Sampling               16        13              11               13.333333
XGBoost Under-Sampling                    18        14              14               15.333333
DecisionTree Over+Under-Sampling          11        18              18               15.666667
LogisticRegression Under-Sampling         17        17              17               17.000000
SGDClassifier Under-Sampling              19        19              19               19.000000
DecisionTree Under-Sampling               20        20              20               20.000000
```

These two tables show the **Precision**, **Recall** and **F1** values and the rankings for each model for the three evaluation metrics, from which several conclusions can be drawn.

1. **Precision**, **Recall**, and **F1** Scores:
    - Best **Precision**: XGBoost No Sampling (0.904348) and RandomForest Over-Sampling (0.929825)
	- Best **Recall**: Models with Under-Sampling tend to have the highest recall, such as DecisionTree Under-Sampling (0.888889)
	- Best **F1** Scores:
	    - RandomForest Over-Sampling (0.851406)
	    - XGBoost Over-Sampling (0.84375)
	    - RandomForest No Sampling (0.834646)
	    - XGBoost No Sampling (0.832)

2. Performance Rankings:
    - Top 3 Models Based on Average Rank:
        - XGBoost Over-Sampling (1.67 average rank)
        - XGBoost No Sampling (2.33 average rank)
        - RandomForest Over-Sampling (2.33 average rank)
    - Lowest Ranked Models:
        - SGDClassifier Under-Sampling (19.00 average rank)
        - DecisionTree Under-Sampling (20.00 average rank)

3. Key Takeaways:
    1.	XGBoost and RandomForest Models:
        - Consistently high performance in **F1**, **Precision-Recall**, and **ROC AUC** metrics.
        - Over-Sampling and No Sampling configurations tend to be the best.
    2.	Under-Sampling Effect:
        - Generally improves **Recall** but significantly lowers **Precision** and **F1** scores.
        - Models like DecisionTree Under-Sampling and LogisticRegression Under-Sampling show poor **F1** scores despite high **Recall**.
    3.	Model Selection:
        - Best Overall: XGBoost Over-Sampling
        - Second Best: XGBoost No Sampling and RandomForest Over-Sampling
        - These models balance high **Precision** and **Recall** effectively, leading to superior **F1** scores and ranking performance.

## Hyperparameter Optimization
With three XGBoost models in the top five best-performing, the next step in improving fraud detection is to choose hyperparameter values which maximize **F1** values and compare them to how the default configuration performs. Information regarding the optimization techniques and their usage can be found in `utils/model_optimization.py`. The three different optimization techniques used in this study are `GridSearchCV`, `RandomizedSearchCV`, and `GASearchCV`. The parameters probed during this study are `learning_rate, n_estimators, gamma, max_depth, reg_alpha, reg_lambda`, which are optimized by cross-validation.

1. GridSearchCV
    - It is an exhaustive search over the specified parameter values for an estimator.
    - The default parameter grid for this algorithm is 
    ```
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 1.],
        "gamma": [0.1, 1., 10., 100.],
        "max_depth": [5, 10, 15],
        "reg_alpha": [0.1, 1., 10.],
        "reg_lambda": [0.1, 1., 10.]
    ```
2. RandomizedSearchCV
    - In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The default parameter grid for this algorithm is the same as for GridSearchCV.

3. GASearchCV
    - Models hyperparameter tuning using evolutionary algorithms. More details regarding this package can be found [here](https://github.com/rodrigo-arenas/Sklearn-genetic-opt).
    - The default parameter grid for this algorithm is
    ```
        "n_estimators": Integer(50, 150),
        "max_depth": Integer(5, 15),
        "learning_rate": Continuous(0.01, 1., distribution="log-uniform"),
        "gamma": Continuous(0.1, 100., distribution="log-uniform"),
        "reg_alpha": Continuous(0.1, 10., distribution="log-uniform"),
        "reg_lambda": Continuous(0.1, 10., distribution="log-uniform"),
    ```

Results running the command `bash run_entire_optimized_analysis.sh XGBoost_HyperParameterOptimization` will run model training and testing using XGBoost classifiers for each of the resampling techniques and all of the optimization algorithm options, plot the results, and rank the models in the same way as before. The results can be found in `results/XGBoost_HyperParameterOptimization/`. Compared to the 30 minute runtime for the default parameters, this study takes approximately 12 hours to run.

### Hyperparameter Training Results
```
                                               Cross_Val   Precision   Recall      F1
XGBoost No Sampling Default                     0.999534    1.0         1.0         1.0
XGBoost No Sampling GASearchCV                  0.999554    1.0         1.0         1.0
XGBoost No Sampling GridSearchCV                0.999534    1.0         0.991597    0.995781
XGBoost No Sampling RandomizedSearchCV          0.999539    1.0         1.0         1.0
XGBoost Over+Under-Sampling Default             0.999631    1.0         1.0         1.0
XGBoost Over+Under-Sampling GASearchCV          0.999665    0.999849    1.0         0.999925
XGBoost Over+Under-Sampling GridSearchCV        0.999615    1.0         1.0         1.0
XGBoost Over+Under-Sampling RandomizedSearchCV  0.999615    1.0         1.0         1.0
XGBoost Over-Sampling Default                   0.99993     1.0         1.0         1.0
XGBoost Over-Sampling GASearchCV                0.999922    0.999995    1.0         0.999997
XGBoost Over-Sampling GridSearchCV              0.99991     1.0         1.0         1.0
XGBoost Over-Sampling RandomizedSearchCV        0.999894    1.0         1.0         1.0
XGBoost Under-Sampling Default                  0.95098     1.0         1.0         1.0
XGBoost Under-Sampling GASearchCV               0.955186    1.0         1.0         1.0
XGBoost Under-Sampling GridSearchCV             0.95098     1.0         1.0         1.0
XGBoost Under-Sampling RandomizedSearchCV       0.95098     1.0         1.0         1.0
```

Here are key observations from the training results:

1. No Sampling
    - High performance with nearly perfect scores for Default, GASearchCV, and RandomizedSearchCV hyperparameter configurations, whereas GridSearchCV has a slightly lower **F1** score compared to others.

2. Over+Under-Sampling
    - Slight improvement in cross-validation scores compared to No Sampling.
    - Default, GridSearchCV, and RandomizedSearchCV all achieve perfect **Precision**, **Recall**, and **F1** scores.
    - GASearchCV achieved the highest cross-validation score, but is the only algorithm to not receive perfect **Precision** and **F1** scores.

3. Over-Sampling
    - Slight improvement in cross-validation scores compared to Over+Under-Sampling.
    - Default, GridSearchCV, and RandomizedSearchCV all achieve perfect **Precision**, **Recall**, and **F1** scores.
    - GASearchCV once again is the only algorithm to not receive perfect **Precision** and **F1** scores, but extremely close.

4. Under-Sampling
    - Significantly lower cross-validation scorescompared to the other sampling methods.
    - All hyperparameter configurations achieve perfect **Precision**, **Recall**, and **F1** scores, with GASearchCV obtaining the highest cross-validation score.

Overall, the models with Over-Sampling achieved the highest cross-validation scores, models with No Sampling and Over+Under-Sampling achieve similar cross-validation scores, and models with Under-Sampling showed significantly lower cross-validation scores.

### Hyperparameter Testing Results
```
---------- Values ----------
                                               Precision   Recall      F1
XGBoost No Sampling Default                     0.904348    0.77037     0.832
XGBoost No Sampling GASearchCV                  0.912281    0.77037     0.835341
XGBoost No Sampling GridSearchCV                0.912281    0.77037     0.835341
XGBoost No Sampling RandomizedSearchCV          0.9375      0.777778    0.850202
XGBoost Over+Under-Sampling Default             0.705128    0.814815    0.756014
XGBoost Over+Under-Sampling GASearchCV          0.709677    0.814815    0.758621
XGBoost Over+Under-Sampling GridSearchCV        0.733333    0.814815    0.77193
XGBoost Over+Under-Sampling RandomizedSearchCV  0.733333    0.814815    0.77193
XGBoost Over-Sampling Default                   0.892562    0.8         0.84375
XGBoost Over-Sampling GASearchCV                0.891667    0.792593    0.839216
XGBoost Over-Sampling GridSearchCV              0.889831    0.777778    0.83004
XGBoost Over-Sampling RandomizedSearchCV        0.867769    0.777778    0.820312
XGBoost Under-Sampling Default                  0.038826    0.911111    0.074478
XGBoost Under-Sampling GASearchCV               0.037975    0.911111    0.07291
XGBoost Under-Sampling GridSearchCV             0.04311     0.903704    0.082293
XGBoost Under-Sampling RandomizedSearchCV       0.04242     0.903704    0.081036

---------- Rankings ----------
                                                F1 Rank  PR AP Rank  ROC AUC < 0.001 Rank  Average Rank
XGBoost No Sampling GASearchCV                     4         2               1                2.333333
XGBoost No Sampling RandomizedSearchCV             1         3               3                2.333333
XGBoost No Sampling GridSearchCV                   5         1               2                2.666667
XGBoost Over-Sampling Default                      2         5               4                3.666667
XGBoost No Sampling Default                        6         4               5                5.000000
XGBoost Over-Sampling GASearchCV                   3         6               6                5.000000
XGBoost Over-Sampling RandomizedSearchCV           8        11               7                8.666667
XGBoost Over+Under-Sampling GridSearchCV           9         7              10                8.666667
XGBoost Over-Sampling GridSearchCV                 7        12               8                9.000000
XGBoost Over+Under-Sampling RandomizedSearchCV    10         8              11                9.666667
XGBoost Over+Under-Sampling Default               12        10               9               10.333333
XGBoost Over+Under-Sampling GASearchCV            11         9              12               10.666667
XGBoost Under-Sampling GridSearchCV               13        15              13               13.666667
XGBoost Under-Sampling GASearchCV                 16        13              14               14.333333
XGBoost Under-Sampling RandomizedSearchCV         14        16              15               15.000000
XGBoost Under-Sampling Default                    15        14              16               15.000000
```

Key trends from the **Precision**, **Recall**, and **F1** values are stated below, as well as model rankings.

1. **Precision**, **Recall**, and **F1** Scores:
    - No Sampling methods generally perform the best, with high **Precision**, **Recall**, and **F1** scores. RandomizedSearchCV slightly outperforms the others within this group.
    - Over-Sampling methods also show strong performance, but not quite as high as No Sampling methods. The Default and GASearchCV configurations perform the best within this group.
    - Over+Under-Sampling methods have lower **Precision** and **F1** scores compared to No Sampling and Over-Sampling methods, though they maintain high **Recall**.
    - Under-Sampling methods are highly ineffective, showing extremely low **Precision** and **F1** scores despite very high **Recall**. Once again, this confirms that Under-Sampling is not a suitable strategy for this dataset and problem.

2. Performance Rankings:
    - No Sampling methods (especially with GASearchCV and RandomizedSearchCV) are the top performers, showing consistent and strong performance across all three metrics.
    - Over-Sampling methods, particularly the Default and GASearchCV configurations, also perform well, though slightly behind the best No Sampling methods.
    - Over+Under-Sampling methods generally show moderate to lower performance, with some configurations like GridSearchCV and Default performing better than others.
    - Under-Sampling methods are the worst performers, with consistently poor rankings across all metrics. This reinforces the previous observation that under-sampling is not effective for this dataset and problem.

In summary, the most effective strategies are No Sampling when combined with any of the optimization techniques studied, but the GASearchCV and RandomizedSearchCV are especially preferred when taking computing time into account. Under-Sampling should be avoided due to its poor performance.
