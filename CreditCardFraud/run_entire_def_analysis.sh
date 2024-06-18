echo "Training and Testing all models and saving results in results/$1"
bash train_test_def_models.sh $1

echo "Combining all individual model output files into single file for plotting training and testing results."
python utils/combine_results.py $1 Train
python utils/combine_results.py $1 Test

echo "Plotting all results."
bash make_all_plots.sh $1
