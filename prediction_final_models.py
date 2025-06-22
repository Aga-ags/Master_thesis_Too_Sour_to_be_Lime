# Import packages
import pandas as pd
import numpy as np
np.random.seed(58)
from keras.models import load_model

# import the NN training script
import NN_training_script as train

# load test data
x_test = pd.read_csv("../data/prepared_data/x_test.csv").values
y_test = pd.read_csv("../data/prepared_data/y_test.csv")

# load inputs for map
x_map = pd.read_csv("../data/prepared_data/.csv").values # change: add the links

# define dfs to save all results into
results_test = y_test
results_map = pd.DataFrame()

# Load the best models and perfrom predictions on test test
# Univariate
results_test = train.load_model_and_predict(model_name = "pH_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_test, 
                                                  df_to_save_output_to = results_test)

results_test = train.load_model_and_predict(model_name = "lime_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_test, 
                                                  df_to_save_output_to = results_test)

# Multivariate
results_test = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_test, 
                                                  df_to_save_output_to = results_test)


results_test = train.load_model_and_predict(model_name = "both_no_3_5_classes_dropout_penatly_05", 
                                                  inputs_prediction = x_test, 
                                                  df_to_save_output_to = results_test)

results_test.to_csv('./tuning_results/test_predictions_best_models.csv', index=False)


# Load the best models and perform the predictions on the entirety of the arable land

results_map = train.load_model_and_predict(model_name = "pH_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map, 
                                                  df_to_save_output_to = results_map)

results_map = train.load_model_and_predict(model_name = "lime_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map, 
                                                  df_to_save_output_to = results_map)

# Multivariate
results_map = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map, 
                                                  df_to_save_output_to = results_map)


results_map = train.load_model_and_predict(model_name = "both_no_3_5_classes_dropout_penatly_05", 
                                                  inputs_prediction = x_map, 
                                                  df_to_save_output_to = results_map)

results_map.to_csv('./tuning_results/test_predictions_map.csv', index=False)
