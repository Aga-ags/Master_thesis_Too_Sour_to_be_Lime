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

# Base model

results_validation = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_val, 
                                                  df_to_save_output_to = results_validation)

# size lambda of the penaltly term

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_dropout_penatly_1",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': train.zero_inflated_mse_loss},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout = True, 
                                    include_lime_ph_penalty = False, 
                                    lambda_penalty = 1)



results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_dropout_penatly_05",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': train.zero_inflated_mse_loss},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout = True, 
                                    include_lime_ph_penalty = False, 
                                    lambda_penalty = 0.5)


results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_dropout_penatly_01",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': train.zero_inflated_mse_loss},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout = True, 
                                    include_lime_ph_penalty = False, 
                                    lambda_penalty = 0.1)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_dropout_penatly_001",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': train.zero_inflated_mse_loss},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout = True, 
                                    include_lime_ph_penalty = False, 
                                    lambda_penalty = 0.01)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_dropout_penatly_0001",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': train.zero_inflated_mse_loss},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout = True, 
                                    include_lime_ph_penalty = False, 
                                    lambda_penalty = 0.001)


results_validation.to_csv('./tuning_results/lambda_parameter_tuning.csv', index=False)
