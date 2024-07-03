export PROJECT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export RESULTS_DIR=$PROJECT_DIR/results
echo "Project working directory set to $PROJECT_DIR"
echo "Results directory set to $RESULTS_DIR"
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
export MLFLOW_EXPERIMENT_NAME="Test Experiment"
