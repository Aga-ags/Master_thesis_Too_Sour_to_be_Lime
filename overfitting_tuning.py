# Import packages
import pandas as pd
import numpy as np
np.random.seed(189)

# import the NN training script
import NN_training_script as train
import tensorflow as tf

# load data
# validation
x_val = pd.read_csv("../data/prepared_data/x_valid.csv").values
y_val = pd.read_csv("../data/prepared_data/y_valid.csv")

# test data
x_test = pd.read_csv("../data/prepared_data/x_test.csv").values
y_test = pd.read_csv("../data/prepared_data/y_test.csv")

# define dfs to save all results into
results_validation = y_val
results_test = y_test

# Load base models

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "pH_no_lime_imputation_from_3_5_classes",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True)

results_validation = train.load_model_and_predict(model_name = "lime_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_val, 
                                                  df_to_save_output_to = results_validation)

results_validation = train.load_model_and_predict(model_name = "both_no_lime_imputation_from_3_5_classes", 
                                                  inputs_prediction = x_val, 
                                                  df_to_save_output_to = results_validation)

# Remaining tuning performed on the base models:
# Counteracting overfitting:
# dropout
results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "pH_no_3_5_classes_dropout",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout=True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "lime_no_3_5_classes_dropout",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout=True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_dropout",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_dropout = True)

# regularization
results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "pH_no_3_5_classes_reg",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_reg = True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "lime_no_3_5_classes_reg",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_reg = True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_reg",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_reg = True)   
# both
results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "pH_no_3_5_classes_reg_dropout",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_reg = True,
                                    include_dropout = True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "lime_no_3_5_classes_reg_dropout",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_reg = True,
                                    include_dropout = True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_3_5_classes_reg_dropout",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True, 
                                    include_reg = True,
                                    include_dropout = True)



results_validation.to_csv('./tuning_results/overfitting_parameter_tuning.csv', index=False)

