TIMEFORMAT='Total Elapsed Time: %1R seconds.'
time {
export MLFLOW_EXPERIMENT_NAME="Default Credit Card Fraud Models"

echo "Training and Testing all models and saving results in results/$1"
python src/mlflow_train_test.py $1 DecisionTree "No Sampling" Default
python src/mlflow_train_test.py $1 DecisionTree "Under-Sampling" Default
python src/mlflow_train_test.py $1 DecisionTree "Over-Sampling" Default
python src/mlflow_train_test.py $1 DecisionTree "Over+Under-Sampling" Default
python src/mlflow_train_test.py $1 LogisticRegression "No Sampling" Default
python src/mlflow_train_test.py $1 LogisticRegression "Under-Sampling" Default
python src/mlflow_train_test.py $1 LogisticRegression "Over-Sampling" Default
python src/mlflow_train_test.py $1 LogisticRegression "Over+Under-Sampling" Default
python src/mlflow_train_test.py $1 RandomForest "No Sampling" Default
python src/mlflow_train_test.py $1 RandomForest "Under-Sampling" Default
python src/mlflow_train_test.py $1 RandomForest "Over-Sampling" Default
python src/mlflow_train_test.py $1 RandomForest "Over+Under-Sampling" Default
python src/mlflow_train_test.py $1 SGDClassifier "No Sampling" Default
python src/mlflow_train_test.py $1 SGDClassifier "Under-Sampling" Default
python src/mlflow_train_test.py $1 SGDClassifier "Over-Sampling" Default
python src/mlflow_train_test.py $1 SGDClassifier "Over+Under-Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "No Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "Under-Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "Over-Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "Over+Under-Sampling" Default

echo "Combining all individual model output files into single file for plotting training and testing results."
python utils/combine_results.py $1 Train
python utils/combine_results.py $1 Test

echo "Plotting all results."
python src/plot_results.py $1 Train
python src/plot_results.py $1 Train --grouping Classifier=DecisionTree
python src/plot_results.py $1 Train --grouping Classifier=LogisticRegression
python src/plot_results.py $1 Train --grouping Classifier=RandomForest
python src/plot_results.py $1 Train --grouping Classifier=SGDClassifier
python src/plot_results.py $1 Train --grouping Classifier=XGBoost
python src/plot_results.py $1 Train --grouping Optimization=Default
python src/plot_results.py $1 Train --grouping Sampling="No Sampling"
python src/plot_results.py $1 Train --grouping Sampling="Under-Sampling"
python src/plot_results.py $1 Train --grouping Sampling="Over-Sampling"
python src/plot_results.py $1 Train --grouping Sampling="Over+Under-Sampling"
python src/plot_results.py $1 Test
python src/plot_results.py $1 Test --grouping Classifier=DecisionTree
python src/plot_results.py $1 Test --grouping Classifier=LogisticRegression
python src/plot_results.py $1 Test --grouping Classifier=RandomForest
python src/plot_results.py $1 Test --grouping Classifier=SGDClassifier
python src/plot_results.py $1 Test --grouping Classifier=XGBoost
python src/plot_results.py $1 Test --grouping Optimization=Default
python src/plot_results.py $1 Test --grouping Sampling="No Sampling"
python src/plot_results.py $1 Test --grouping Sampling="Under-Sampling"
python src/plot_results.py $1 Test --grouping Sampling="Over-Sampling"
python src/plot_results.py $1 Test --grouping Sampling="Over+Under-Sampling"

echo "Ranking all of the models."
python src/rank_models.py $1
}
