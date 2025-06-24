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
x_map_10 = pd.read_csv("../data/prepared_data/x_map_10.csv").values 
x_map_60 = pd.read_csv("../data/prepared_data/x_map_60.csv").values
coordinates = pd.read_csv("../data/prepared_data/coordinates_map.csv")

# define dfs to save all results into
results_test = y_test
results_map_10 = coordinates
results_map_60 = coordinates

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


results_test = train.load_model_and_predict(model_name = "both_no_3_5_classes_penatly_05_correction1", 
                                                  inputs_prediction = x_test, 
                                                  df_to_save_output_to = results_test)

results_test.to_csv('./tuning_results/test_predictions_best_models.csv', index=False)


# Load the best models and perform the predictions on the entirety of the arable land

results_map_10 = train.load_model_and_predict(model_name = "pH_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map_10, 
                                                  df_to_save_output_to = results_map_10)

results_map_10 = train.load_model_and_predict(model_name = "lime_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map_10, 
                                                  df_to_save_output_to = results_map_10)

# Multivariate
results_map_10 = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map_10, 
                                                  df_to_save_output_to = results_map_10)


results_map_10 = train.load_model_and_predict(model_name = "both_no_3_5_classes_penatly_05_correction1", 
                                                  inputs_prediction = x_map_10, 
                                                  df_to_save_output_to = results_map_10)

results_map_10.to_csv('./tuning_results/predictions_map_10.csv', index=False)

results_map_60 = train.load_model_and_predict(model_name = "pH_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map_60, 
                                                  df_to_save_output_to = results_map_60)

results_map_60 = train.load_model_and_predict(model_name = "lime_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map_60, 
                                                  df_to_save_output_to = results_map_60)

# Multivariate
results_map_60 = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_map_60, 
                                                  df_to_save_output_to = results_map_60)


results_map_60 = train.load_model_and_predict(model_name = "both_no_3_5_classes_penatly_05_correction1", 
                                                  inputs_prediction = x_map_60, 
                                                  df_to_save_output_to = results_map_60)

results_map_60.to_csv('./tuning_results/predictions_map_60.csv', index=False)