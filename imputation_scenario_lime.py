# Import packages
import pandas as pd
import numpy as np
np.random.seed(189)

# import the NN training script
import NN_training_script as train

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


# train the univariate NN with lime as output under the various imputation strategies

train.perform_moodel_training_with_tuning(imputation_scenario = "full_imp", 
                                    model_name = "lime_full_imp",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "lime_no_lime_imputation_from_3_5_classes",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes", 
                                    model_name = "lime_no_lime_imputation_from_lime_classes",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "lime_no_lime_imputation",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = True)

# The same but without weights
train.perform_moodel_training_with_tuning(imputation_scenario = "full_imp", 
                                    model_name = "lime_full_imp_no_weights",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "lime_no_lime_imputation_from_3_5_classes_no_weights",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes", 
                                    model_name = "lime_no_lime_imputation_from_lime_classes_no_weights",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "lime_no_lime_imputation_no_weights",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': 'mse'},
                                    include_weigths = False)

results_validation.to_csv('./tuning_results/lime_imputation_scenario_validation.csv', index=False)
