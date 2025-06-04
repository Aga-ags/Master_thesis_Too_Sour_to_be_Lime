# Experiment: remove 2 node option, increase epochs and patience, more layer options, remove relu activation on last layer

import pandas as pd
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

np.random.seed(189)
import tensorflow as tf
import keras
from keras import layers
from keras import ops
from keras.layers import Dropout
from keras import regularizers
import keras_tuner as kt
import pickle

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

# custom objective for zero - inflation

class CustomMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="zero_inflated_mse", **kwargs)

        # stores cumulative sum of squares of true zero samples
        self.sum_zero = self.add_weight(name="sum_zero", initializer="zeros")
        # stores the number of zero samples
        self.count_zero = self.add_weight(name="count_zero", initializer="zeros")
        # stores cumulative sum of squares of samples greater than zero 
        self.sum_gt_zero = self.add_weight(name="sum_gt_zero", initializer="zeros")
        self.count_gt_zero = self.add_weight(name="count_gt_zero", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight = None):
        # Create masks
        mask_zero = ops.equal(y_true, 0)
        mask_gt_zero = ops.greater(y_true, 0)

        # MSE for zero values
        y_true_zero = tf.boolean_mask(y_true, mask_zero)
        y_pred_zero = tf.boolean_mask(y_pred, mask_zero)
        se_zero = ops.square(y_true_zero - y_pred_zero)
        self.sum_zero.assign_add(ops.sum(se_zero))
        self.count_zero.assign_add(ops.cast(ops.shape(se_zero)[0], "float32"))

        # MSE for values > 0
        y_true_gt_zero = tf.boolean_mask(y_true, mask_gt_zero)
        y_pred_gt_zero = tf.boolean_mask(y_pred, mask_gt_zero)
        se_gt_zero = ops.square(y_true_gt_zero - y_pred_gt_zero)
        self.sum_gt_zero.assign_add(ops.sum(se_gt_zero))
        self.count_gt_zero.assign_add(ops.cast(ops.shape(se_gt_zero)[0], "float32"))


    def result(self):
        mse_zero = tf.cond(
            self.count_zero > 0,
            lambda: self.sum_zero / self.count_zero,
            lambda: tf.constant(0.0)
        )

        mse_gt_zero = tf.cond(
            self.count_gt_zero > 0,
            lambda: self.sum_gt_zero / self.count_gt_zero,
            lambda: tf.constant(0.0)
        )

        return (mse_zero + mse_gt_zero) / 2

    def reset_state(self):
        self.sum_zero.assign(0.0)
        self.count_zero.assign(0.0)
        self.sum_gt_zero.assign(0.0)
        self.count_gt_zero.assign(0.0)


# Define the model building function
def build_model(hp, number_outputs, x_train, loss_function, include_dropout, reg):
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(x_train.shape[1],)))
    if include_dropout:
        model.add(Dropout(0.2))

    # Tune the number of hidden layers: 1 to 3
    for i in range(hp.Int("num_layers", 0, 4)):
        model.add(
            layers.Dense(
                units=hp.Choice(f"units_{i}", values=[16, 32, 64, 128]),
                activation=hp.Choice("activation", values=["relu", "tanh", "elu", "silu"]),
                kernel_regularizer= reg 
            ))
        if include_dropout:
            model.add(Dropout(0.2))
    
    model.add(layers.Dense(
                units=hp.Choice(f"units_last_hidden_layer", values=[2, 16, 32, 64, 128]),
                activation=hp.Choice("activation_last_hidden_layer", values=["relu", "tanh", "elu", "silu"])))
    if include_dropout:
        model.add(Dropout(0.2))
        
    
    # Output layer for regression
    model.add(layers.Dense(number_outputs, activation=hp.Choice("activation_last_layer", values=["linear", "relu"])))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss=loss_function,
        metrics=["mae", "mse" #, CustomMetric()
                 ]
    )
    
    return model

# define model training and parameter tuning function
def perform_moodel_training_with_tuning(imputation_scenario, model_name, include_weigths, include_dropout, list_of_dependent_var, objective, loss_function, reg):
    x_train = pd.read_csv("../data/x_train_" + imputation_scenario + ".csv").values
    y_train = pd.read_csv("../data/y_train_" + imputation_scenario + ".csv")

    # sample wrights
    sample_weights = pd.read_csv("../data/sample_weights_" + imputation_scenario + ".csv").values.flatten()

    
    training_y = y_train[list_of_dependent_var].values
    validation_y = y_val[list_of_dependent_var].values

    # Initialize the tuner
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp,  number_outputs = len(list_of_dependent_var), x_train = x_train, loss_function = loss_function, include_dropout = include_dropout, reg = reg),
        objective = objective, # org val_mae
        max_trials=100,
        executions_per_trial=1,
        directory="keras_tuner_logs",
        project_name = model_name 
    )

    # Define training arguments
    search_args = {
        "x": x_train,
        "y": training_y,
        "validation_data": (x_val, validation_y),
        "epochs": 200,
        "batch_size": 32,
        "callbacks":[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
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
    best_model.save("./tuning_results/"+ model_name + ".keras")

    # Evaluate best model on validation data
    loss, mae, mse = best_model.evaluate(x_val, validation_y)
    rmse = np.sqrt(mse)
    
    print(f"Metrics of best model on validation set")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    #print(f"Zero inflated MSE:  {zero_infalted_mse:.4f}")

    # save the predicted values for vizualization
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    if len(list_of_dependent_var) == 1:
        results_validation[model_name] = y_pred_val
        results_test[model_name] = y_pred_test

    elif len(list_of_dependent_var) == 2:
        results_validation[model_name + ".pH"] = y_pred_val[:, [0]]
        results_validation[model_name + ".lime"] = y_pred_val[:, [1]]
        results_test[model_name + ".pH"] = y_pred_test[:, [0]]
        results_test[model_name + ".lime"] = y_pred_test[:, [1]]

    else:
        print("The output was not saved due to unexpeced number of variables")

    # save best model training history
    history = best_model.fit(
    x_train,
    training_y,
    validation_data=(x_val, validation_y),
    epochs=200,
    batch_size=32,
    sample_weight=sample_weights if include_weigths else None,
    callbacks=[keras.callbacks.EarlyStopping(monitor= objective , patience=10, restore_best_weights=True)]
)
    with open("./tuning_results/" + model_name + "_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

# single output lime: 
# everything per lime imputation scenario 

perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65_0_2", 
                                    model_name = "lime_imp_and_lime_65_0_2_l1",
                                    include_weigths = True,
                                    list_of_dependent_var = ["lime"],
                                    objective= "val_mse", 
                                    loss_function = "mse",
                                    include_dropout = True, 
                                    reg = regularizers.l1(1e-2))


perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65_0_2", 
                                    model_name = "lime_imp_and_lime_65_0_2_l2",
                                    include_weigths = True,
                                    list_of_dependent_var = ["lime"],
                                    objective= "val_mse", 
                                    loss_function = "mse",
                                    include_dropout = True, 
                                    reg = regularizers.l2(1e-2))

results_validation.to_csv('./tuning_results/validation_with_regulralization.csv', index=False)
results_test.to_csv('./tuning_results/test_with_regulralization.csv', index=False)







