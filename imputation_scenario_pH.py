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

# train the univariate NN with pH as output under the various imputation strategies

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "full_imp", 
                                    model_name = "pH_full_imp_no_weights",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = False)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_site", 
                                    model_name = "pH_no_site_no_weights",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = False)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_site_H2O", 
                                    model_name = "pH_no_site_H2O_no_weights",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = False)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "full_imp", 
                                    model_name = "pH_full_imp",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_site", 
                                    model_name = "pH_no_site",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True)

results_validation, results_test = train.perform_moodel_training_with_tuning(imputation_scenario = "no_site_H2O", 
                                    model_name = "pH_no_site_H2O",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    results_validation=results_validation,
                                    results_test=results_test,
                                    include_weigths = True)



results_validation.to_csv('./tuning_results/pH_imputation_scenario_validation.csv', index=False)



