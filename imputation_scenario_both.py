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


#train the multivariate NN with pH and lime as output under the various imputation strategies, with and without weights

train.perform_moodel_training_with_tuning(imputation_scenario = "full_imp", 
                                    model_name = "both_full_imp",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_site_H2O", 
                                    model_name = "both_no_site_H2O",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_site", 
                                    model_name = "both_no_site",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "both_no_lime_imputation",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes", 
                                    model_name = "both_no_lime_imputation_from_lime_classes",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_lime_imputation_from_3_5_classes",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_imputation", 
                                    model_name = "both_no_imputation",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_site", 
                                    model_name = "both_no_lime_imputation_site",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes_H2O_site", 
                                    model_name = "both_no_lime_imputation_from_lime_classes_H2O_site",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes_site", 
                                    model_name = "both_no_lime_imputation_from_lime_classes_site",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes_H2O_site", 
                                    model_name = "both_no_lime_imputation_from_3_5_classes_H2O_site",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes_site", 
                                    model_name = "both_no_lime_imputation_from_3_5_classes_site",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = True)

# multivariate, but no weights
train.perform_moodel_training_with_tuning(imputation_scenario = "full_imp", 
                                    model_name = "both_full_imp_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_site_H2O", 
                                    model_name = "both_no_site_H2O_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_site", 
                                    model_name = "both_no_site_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "both_no_lime_imputation_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes", 
                                    model_name = "both_no_lime_imputation_from_lime_classes_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
                                    model_name = "both_no_lime_imputation_from_3_5_classes_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_imputation", 
                                    model_name = "both_no_imputation_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_site", 
                                    model_name = "both_no_lime_imputation_site_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes_H2O_site", 
                                    model_name = "both_no_lime_imputation_from_lime_classes_H2O_site_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_lime_classes_site", 
                                    model_name = "both_no_lime_imputation_from_lime_classes_site_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes_H2O_site", 
                                    model_name = "both_no_lime_imputation_from_3_5_classes_H2O_site_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)

train.perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes_site", 
                                    model_name = "both_no_lime_imputation_from_3_5_classes_site_no_weights",
                                    list_of_outputs = ["pH", "lime"],
                                    loss_function = {'pH': 'mse', 'lime': 'mse'},
                                    include_weigths = False)


results_validation.to_csv('./tuning_results/both_imputation_scenario_validation.csv', index=False)

