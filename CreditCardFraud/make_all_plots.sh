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
