echo "Running preprocessing, model training, and model testing with output in results/$1"
python src/preprocessing.py $1
python src/model_training.py $1
python src/model_testing.py $1
