import pandas as pd
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

np.random.seed(189)
import tensorflow as tf
import keras
from keras import Model, Input
from keras import ops
from keras.layers import Dense, Dropout
import keras_tuner as kt
import pickle

# load data
# validation
x_val = pd.read_csv("../data/x_valid.csv").values
y_val = pd.read_csv("../data/y_valid.csv")

# transform the validation y to dictionary, since that is the format needed for evaluation
validation_y = {
    "pH": y_val["pH"].values,
    "lime": y_val["lime"].values
    }

# test data
x_test = pd.read_csv("../data/x_test.csv").values
y_test = pd.read_csv("../data/y_test.csv")

# define dfs to save all results into
results_validation = y_val
results_test = y_test

# custom objective for zero - inflation
class zero_inflated_mse(keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric 
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

# Build the neural network architecture 
def build_model(hp, outputs, x_train, loss_function, include_dropout):
    inputs = Input(shape=(x_train.shape[1],), name="input")
    x = inputs

    if include_dropout:
        x = Dropout(0.2)(x)

    # Add hidden layers
    for i in range(hp.Int("num_layers", 0, 5)): # hp functions define the search space of parameter tunning
        x = Dense(
            units=hp.Choice(f"units_{i}", values=[2, 8, 16, 32, 64, 128]),
            activation=hp.Choice("activation", values=["relu", "tanh", "elu", "silu"])
        )(x)
        if include_dropout:
            x = Dropout(0.2)(x)

    # Output layers with names
    if "pH" in outputs:
        output_ph = Dense(1, name="pH", activation="linear")(x)
    if "lime" in outputs:    
        output_lime = Dense(1, name="lime", activation=hp.Choice("activation_last_layer", values=["linear", "relu"]))(x)

    model = Model(inputs=inputs, outputs={"pH": output_ph, "lime": output_lime})

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss=loss_function, #if a dictionary, by default it adds the two 
        metrics={"pH": ["mae", "mse"], "lime": ["mae", "mse", zero_inflated_mse()]}
    )

    return model

# define perform model training with parameter tunning
def perform_moodel_training_with_tuning(imputation_scenario, model_name, include_weigths, include_dropout, list_of_dependent_var, objective, loss_function):

    # load training data 
    x_train = pd.read_csv("../data/x_train_" + imputation_scenario + ".csv").values
    y_train = pd.read_csv("../data/y_train_" + imputation_scenario + ".csv")

    # transfrom the training y values to dictionaries for evaluation
    training_y = {
        "pH": y_train["pH"].values,
        "lime": y_train["lime"].values
    }

    # Initialize the tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp,  outputs = list_of_dependent_var, x_train = x_train, loss_function = loss_function, include_dropout = include_dropout),
        objective = objective, # org val_mae
        max_trials=10, # 50?
        executions_per_trial=1,
        directory="keras_tuner_logs",
        project_name = model_name 
    )

    # Define training arguments
    search_args = {
        "x": x_train,
        "y": training_y,
        "validation_data": (x_val, validation_y),
        "epochs": 10,
        "batch_size": 32,
        "callbacks":[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    }

    # Add sample_weight only if include_weights is True
    if include_weigths:
        # load sample weights
        sample_weights = pd.read_csv("../data/sample_weights_" + imputation_scenario + ".csv").values.flatten()
        # Add them to search arguments
        search_args["sample_weight"] = {
        "pH": sample_weights,
        "lime": sample_weights
    }

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
    metrics = best_model.evaluate(x_val, validation_y, return_dict=True)
    print("Best model metrics")
    print(metrics)
    
    # save the predicted values for vizualization
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    if len(list_of_dependent_var) == 1:
        results_validation[model_name] = y_pred_val
        results_test[model_name] = y_pred_test

    elif len(list_of_dependent_var) == 2:
        results_validation[model_name + ".pH"] = y_pred_val["pH"]
        results_validation[model_name + ".lime"] = y_pred_val["pH"]
        results_test[model_name + ".pH"] = y_pred_test["lime"]
        results_test[model_name + ".lime"] = y_pred_test["lime"]

    else:
        print("The output was not saved due to unexpeced number of variables")

    print("output names: ")
    print(best_model.output_names)
    # save best model training history
    history = best_model.fit(
    x_train,
    training_y,
    validation_data=(x_val, validation_y),
    epochs=200,
    batch_size=32,
    sample_weight={"pH": sample_weights, "lime": sample_weights} if include_weigths else None,
    callbacks=[keras.callbacks.EarlyStopping(monitor= 'val_loss' , patience=10, restore_best_weights=True)]
)
    with open("./tuning_results/" + model_name + "_history.pkl", "wb") as f:
        pickle.dump(history.history, f)


perform_moodel_training_with_tuning(imputation_scenario = "pH_imp_and_lime_65_0_2", 
                                    model_name = "both_funtional_api_try4",
                                    include_weigths = False,
                                    list_of_dependent_var = ["pH","lime"],
                                    objective=[kt.Objective("val_pH_mse", direction="min"), kt.Objective("val_lime_mse", direction="min")],
                                    loss_function = {'pH': 'mse', 'lime': "mse"},
                                    include_dropout = True)



results_validation.to_csv('./tuning_results/functional_api_val.csv', index=False)
results_test.to_csv('./tuning_results/functional_api_test.csv', index=False)





