import pandas as pd
import numpy as np
np.random.seed(189)
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from sklearn.metrics import r2_score

# load data
# validation
x_val = pd.read_csv("../data/x_valid.csv").values
y_val = pd.read_csv("../data/y_valid.csv")

# test data
x_test = pd.read_csv("../data/x_test.csv").values
y_test = pd.read_csv("../data/y_test.csv")


# define dfs to save all results into
results_validation = y_val
results_test = y_test


# Define the model building function
def build_model(hp, number_outputs, x_train):
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(x_train.shape[1],)))

    # Tune the number of hidden layers: 1 to 3
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Choice(f"units_{i}", values=[16, 32, 64, 128]),
                activation=hp.Choice("activation", values=["relu", "tanh"])
            )
        )
    
    # Output layer for regression
    model.add(layers.Dense(number_outputs))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
        metrics=["mae", "mse"]
    )
    
    return model

# define model training and parameter tuning function
def perform_moodel_training_with_tuning(imputation_scenario, model_name, include_weigths, list_of_dependent_var):
    x_train = pd.read_csv("../data/x_train_" + imputation_scenario + ".csv").values
    y_train = pd.read_csv("../data/y_train_" + imputation_scenario + ".csv")

    # sample wrights
    sample_weights = pd.read_csv("../data/sample_weights_" + imputation_scenario + ".csv").values.flatten()

    
    training_y = y_train[list_of_dependent_var].values
    validation_y = y_val[list_of_dependent_var].values

    # Initialize the tuner
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp,  number_outputs = len(list_of_dependent_var), x_train = x_train),
        objective="val_mse", # best model is selected based on lowest MAE in the validation data
        max_trials=10,
        executions_per_trial=1,
        directory="keras_tuner_logs",
        project_name = model_name 
    )

    # Define training arguments
    search_args = {
        "x": x_train,
        "y": training_y,
        "validation_data": (x_val, validation_y),
        "epochs": 50,
        "batch_size": 32,
        "callbacks": [keras.callbacks.EarlyStopping(patience=5)]
    }

    # Add sample_weight only if include_weights is True
    if include_weigths:
        search_args["sample_weight"] = sample_weights

    # Call tuner.search with unpacked arguments
    tuner.search(**search_args)

    # print search summary
    tuner.search_space_summary()

    # Get the parameters of best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # Print best hyperparameters
    print("Best Hyperparameters:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    # Save the best model
    best_model.save("./tuning_results/"+ model_name + ".h5")

    # Evaluate best model on validation data
    loss, mae, mse = best_model.evaluate(x_val, validation_y)
    rmse = np.sqrt(mse)
    
    print(f"Metrics of best model on validation set")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # save the predicted values for vizualization
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    if len(list_of_dependent_var) == 1:
        results_validation[model_name] = y_pred_val
        results_test[model_name] = y_pred_test

    elif len(list_of_dependent_var) == 2:
        results_validation[model_name + "pH"] = y_pred_val[:, [0]]
        results_validation[model_name + "lime"] = y_pred_val[:, [1]]
        results_test[model_name + "pH"] = y_pred_test[:, [0]]
        results_test[model_name + "lime"] = y_pred_test[:, [1]]

    else:
        print("The output was not saved due to unexpeced number of variables")


# # single output pH: 
# perform_moodel_training_with_tuning(imputation_scenario= "only_H20", 
#                                     model_name = "pH_only_H2O",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["pH"])

# perform_moodel_training_with_tuning(imputation_scenario= "pH_imp", 
#                                     model_name = "pH_pH_imp",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["pH"])

# perform_moodel_training_with_tuning(imputation_scenario=  "H20_and_lime_65",
#                                     model_name = "pH_H20_and_lime_65",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["pH"])


# perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65", 
#                                     model_name = "pH_imp_and_lime_65",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["pH"])

# perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65_0_2", 
#                                     model_name = "pH_imp_and_lime_65_0_2",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["pH"])

# perform_moodel_training_with_tuning(imputation_scenario = "full_imp",
#                                     model_name = "pH_full_imp",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["pH"])

# # Export results to CSV
# results_validation.to_csv('./tuning_results/predictions_validation_pH.csv', index=False)
# results_test.to_csv('./tuning_results/predictions_test_pH.csv', index=False)


# single output lime: 
# perform_moodel_training_with_tuning(imputation_scenario= "only_H20", 
#                                     model_name = "lime_only_H2O",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["lime"])

# perform_moodel_training_with_tuning(imputation_scenario= "pH_imp", 
#                                     model_name = "lime_pH_imp",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["lime"])

# perform_moodel_training_with_tuning(imputation_scenario=  "H20_and_lime_65",
#                                     model_name = "lime_H20_and_lime_65",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["lime"])


# perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65", 
#                                     model_name = "lime_imp_and_lime_65",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["lime"])

# perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65_0_2", 
#                                     model_name = "lime_imp_and_lime_65_0_2",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["lime"])

# perform_moodel_training_with_tuning(imputation_scenario = "full_imp",
#                                     model_name = "lime_full_imp",
#                                     include_weigths = True,
#                                     list_of_dependent_var = ["lime"])


# # Export results to CSV
# results_validation.to_csv('./tuning_results/predictions_validation_lime.csv', index=False)
# results_test.to_csv('./tuning_results/predictions_test_lime.csv', index=False)


# two output: 
perform_moodel_training_with_tuning(imputation_scenario= "only_H20", 
                                    model_name = "both_only_H2O",
                                    include_weigths = True,
                                    list_of_dependent_var = ["pH", "lime"])

perform_moodel_training_with_tuning(imputation_scenario= "pH_imp", 
                                    model_name = "both_pH_imp",
                                    include_weigths = True,
                                    list_of_dependent_var = ["pH","lime"])

perform_moodel_training_with_tuning(imputation_scenario=  "H20_and_lime_65",
                                    model_name = "both_H20_and_lime_65",
                                    include_weigths = True,
                                    list_of_dependent_var = ["pH","lime"])


perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65", 
                                    model_name = "both_imp_and_lime_65",
                                    include_weigths = True,
                                    list_of_dependent_var = ["pH","lime"])

perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65_0_2", 
                                    model_name = "both_imp_and_lime_65_0_2",
                                    include_weigths = True,
                                    list_of_dependent_var = ["pH","lime"])

perform_moodel_training_with_tuning(imputation_scenario = "full_imp",
                                    model_name = "both_full_imp",
                                    include_weigths = True,
                                    list_of_dependent_var = ["pH","lime"])


# Export results to CSV
results_validation.to_csv('./tuning_results/predictions_validation_both.csv', index=False)
results_test.to_csv('./tuning_results/predictions_test_both.csv', index=False)

