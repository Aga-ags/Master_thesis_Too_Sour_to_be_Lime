# Import packages
import pandas as pd
import numpy as np
np.random.seed(153)
from keras.models import load_model

# import the NN training script
import NN_training_script as train

# load data
# validation
x_val = pd.read_csv("../data/prepared_data/x_valid.csv").values
y_val = pd.read_csv("../data/prepared_data/y_valid.csv")

# test data
x_test = pd.read_csv("../data/prepared_data/x_test.csv").values
y_test = pd.read_csv("../data/prepared_data/y_test.csv")

# define dfs to save results into
results_validation = y_val
results_test = y_test


results_validation = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_val, 
                                                  df_to_save_output_to = results_validation)


results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_penatly_1_squared",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_lime_ph_penalty = True, 
                                    lambda_penalty = 1)


results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_penatly_01_squared",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_lime_ph_penalty = True, 
                                    lambda_penalty = 0.1)


results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_penatly_001_squared",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_lime_ph_penalty = True, 
                                    lambda_penalty = 0.01)



results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_penatly_0001_squared",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_lime_ph_penalty = True, 
                                    lambda_penalty = 0.001)



results_validation.to_csv('./tuning_results/test2.csv', index=False)