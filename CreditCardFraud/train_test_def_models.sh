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
