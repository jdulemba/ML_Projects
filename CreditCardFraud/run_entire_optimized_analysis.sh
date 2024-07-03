TIMEFORMAT='Total Elapsed Time: %1R seconds.'
time {
export MLFLOW_EXPERIMENT_NAME="Optimized XGBoost Credit Card Fraud Models"

echo "Training and Testing all models and saving results in results/$1"
python src/mlflow_train_test.py $1 XGBoost "No Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "No Sampling" GridSearchCV
python src/mlflow_train_test.py $1 XGBoost "No Sampling" RandomizedSearchCV
python src/mlflow_train_test.py $1 XGBoost "No Sampling" GASearchCV
python src/mlflow_train_test.py $1 XGBoost "Under-Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "Under-Sampling" GridSearchCV
python src/mlflow_train_test.py $1 XGBoost "Under-Sampling" RandomizedSearchCV
python src/mlflow_train_test.py $1 XGBoost "Under-Sampling" GASearchCV
python src/mlflow_train_test.py $1 XGBoost "Over-Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "Over-Sampling" GridSearchCV
python src/mlflow_train_test.py $1 XGBoost "Over-Sampling" RandomizedSearchCV
python src/mlflow_train_test.py $1 XGBoost "Over-Sampling" GASearchCV
python src/mlflow_train_test.py $1 XGBoost "Over+Under-Sampling" Default
python src/mlflow_train_test.py $1 XGBoost "Over+Under-Sampling" GridSearchCV
python src/mlflow_train_test.py $1 XGBoost "Over+Under-Sampling" RandomizedSearchCV
python src/mlflow_train_test.py $1 XGBoost "Over+Under-Sampling" GASearchCV

echo "Combining all individual model output files into single file for plotting training and testing results."
python utils/combine_results.py $1 Train
python utils/combine_results.py $1 Test

echo "Plotting all results."
python src/plot_results.py $1 Train
python src/plot_results.py $1 Train --grouping Classifier=XGBoost
python src/plot_results.py $1 Train --grouping Optimization=Default
python src/plot_results.py $1 Train --grouping Optimization=GridSearchCV
python src/plot_results.py $1 Train --grouping Optimization=RandomizedSearchCV
python src/plot_results.py $1 Train --grouping Optimization=GASearchCV
python src/plot_results.py $1 Train --grouping Sampling="No Sampling"
python src/plot_results.py $1 Train --grouping Sampling="Under-Sampling"
python src/plot_results.py $1 Train --grouping Sampling="Over-Sampling"
python src/plot_results.py $1 Train --grouping Sampling="Over+Under-Sampling"
python src/plot_results.py $1 Test
python src/plot_results.py $1 Test --grouping Classifier=XGBoost
python src/plot_results.py $1 Test --grouping Optimization=Default
python src/plot_results.py $1 Test --grouping Optimization=GridSearchCV
python src/plot_results.py $1 Test --grouping Optimization=RandomizedSearchCV
python src/plot_results.py $1 Test --grouping Optimization=GASearchCV
python src/plot_results.py $1 Test --grouping Sampling="No Sampling"
python src/plot_results.py $1 Test --grouping Sampling="Under-Sampling"
python src/plot_results.py $1 Test --grouping Sampling="Over-Sampling"
python src/plot_results.py $1 Test --grouping Sampling="Over+Under-Sampling"

echo "Ranking all of the models."
python src/rank_models.py $1
}
