import pandas as pd
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

np.random.seed(189)
import tensorflow as tf
import keras
from keras import Model, Input, regularizers, activations
from keras.models import load_model
from keras.layers import Dense, Dropout, Concatenate, Activation
from keras.utils import get_custom_objects
import keras.backend as K
import matplotlib.pyplot as plt
import keras_tuner as kt
import json

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

def custom_relu(x):
    return tf.where(x > - 0.5062935, x, tf.constant(- 0.5062935, dtype=x.dtype))

# custom objective for zero - inflation
def zero_inflated_mse_loss(y_true, y_pred):
    # Due to standarization lime = 0 is now roughly equal to - 0.5062935, which is not a nice intiger to compare to, therefore we set a tolerance level
    tolerance = 1e-4
    mask_zero = tf.abs(y_true + 0.5062935) < tolerance # values from -0.0001 to 0.0001 are considered 0
    mask_gt_zero = y_true > (-0.5062935 + tolerance) # values larger than -0.5063935 are considered larger than 0

     # MSE for zero values
    y_true_zero = tf.boolean_mask(y_true, mask_zero)
    y_pred_zero = tf.boolean_mask(y_pred, mask_zero)

    mse_zero = tf.cond(
        tf.size(y_true_zero) > 0,
        lambda: tf.reduce_mean(tf.square(y_true_zero - y_pred_zero)),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

    # MSE for values > 0
    y_true_gt_zero = tf.boolean_mask(y_true, mask_gt_zero)
    y_pred_gt_zero = tf.boolean_mask(y_pred, mask_gt_zero)

    mse_gt_zero = tf.cond(
        tf.size(y_true_gt_zero) > 0,
        lambda: tf.reduce_mean(tf.square(y_true_gt_zero - y_pred_gt_zero)),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

    return (mse_zero + mse_gt_zero) / 2.0

# def lime_ph_penalty_loss(lambda_penalty=1.0):
#     def loss(y_true, y_pred):
#         output_pH = y_pred[:, 0]
#         output_lime = y_pred[:, 1]

#         # Mask where pH < 6.5
#         mask = tf.cast(tf.less(output_pH, -0.115569), tf.float32)  # -0.115568 is equvalent to pH 6.5 post standarization
#         # Compute penalty
#         masked_lime = output_lime * mask # extract lime values for which predicted pH is below 6.5
#         mask_sum = tf.reduce_sum(mask) # get number of cases where pH < 6.5 occurs
#         epsilon = 1e-6
#         penalty = tf.reduce_sum(tf.abs(masked_lime + 0.5062935)) / (mask_sum + epsilon)

#         return lambda_penalty * penalty
#     return loss

def lime_ph_penalty_loss(lambda_penalty=1.0):
    def loss(y_true, y_pred):
        output_pH = y_pred[:, 0]
        output_lime = y_pred[:, 1]

        # Mask where pH < 6.5
        mask = tf.cast(tf.less(output_pH, -0.115569), tf.float32)  # -0.115568 is equvalent to pH 6.5 post standarization
        # Compute penalty
        masked_lime = output_lime * mask # extract lime values for which predicted pH is below 6.5
        penalty = tf.reduce_mean(tf.square(masked_lime + 0.5062935))

        return lambda_penalty * penalty
    return loss

def create_objectives_from_loss(loss_function, list_of_outputs, include_lime_ph_penalty):
    if len(list_of_outputs) == 1: 
        objectives_list = [kt.Objective(f"val_loss", direction="min")]
    elif len(list_of_outputs) == 2:     
        objectives_list = [kt.Objective(f"val_{key}_loss", direction="min") for key, loss in loss_function.items()]
        if include_lime_ph_penalty == True: 
            objectives_list.append(kt.Objective("val_combined_output_loss", direction="min"))
    else: 
        print("Unexpected number of outputs in the objective creation function")
        objectives_list = None
    return objectives_list

# Build the neural network architecture 
def build_model(hp, list_of_outputs, x_train, loss_function, include_dropout = False, include_lime_ph_penalty = False, lambda_penalty = 0.001, include_reg = False):
    inputs = Input(shape=(x_train.shape[1],), name="input")
    x = inputs

    # Set up regularizer once if using regularization
    if include_reg:
        reg_type = hp.Choice("regularization_type", ["l1", "l2", "l1_l2"])
        reg_strength = hp.Choice("reg_strength", [1e-4, 1e-3, 1e-2])
        if reg_type == "l1":
            kernel_regularizer = regularizers.l1(reg_strength)
        elif reg_type == "l2":
            kernel_regularizer = regularizers.l2(reg_strength)
        else:
            kernel_regularizer = regularizers.l1_l2(l1=reg_strength, l2=reg_strength)
    else:
        kernel_regularizer = None

    # set dropout strength
    if include_dropout:
        dropout_strength = hp.Choice("dropout_strength", [0.2, 0.3, 0.4, 0.5])

    # Activation can be consistent across layers (or randomized per layer if you prefer)
    activation = hp.Choice("activation", values=["relu", "tanh", "elu", "silu"])

     # Build hidden layers
    for i in range(hp.Int("num_layers", 1, 5)):
        units = hp.Choice(f"units_{i}", values=[16, 32, 64, 128])
        x = Dense(units=units, activation=activation,
                  kernel_regularizer=kernel_regularizer)(x)
        if include_dropout:
            x = Dropout(dropout_strength)(x)

    # From here the model definition depends on which outputs where indicated in the list_of_outputs
    output_dict = {}
    metric_dict = {}

    if "pH" in list_of_outputs:
        output_ph = Dense(1, name="pH", activation="linear")(x) # add final layer of NN
        output_dict["pH"] = output_ph # add pH to outputs of the model
        metric_dict["pH"] = ["mae", "mse"] # add measures that make sense for pH to the metrics 

    if "lime" in list_of_outputs:
        #output_lime = Dense(1, name="lime", activation=Activation(lambda x: keras.activations.relu(x, threshold=-0.5062935)))(x) # activation function: 
        output_lime = Dense(1, name="lime", activation=custom_relu)(x)
        output_dict["lime"] = output_lime
        metric_dict["lime"] = ["mae", "mse"]

    # remove
    if "pH" in list_of_outputs and "lime" in list_of_outputs and include_lime_ph_penalty == True: 
        combined_outputs = Concatenate(name="combined_output")([output_ph, output_lime])
        output_dict["combined_output"] = combined_outputs
        metric_dict["combined_output"] = []
        loss_function["combined_output"] = lime_ph_penalty_loss(lambda_penalty)

    # Create the model instance   
    model = Model(inputs=inputs, outputs=output_dict)

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss= loss_function, #if a dictionary, by default it adds the two 
        metrics = metric_dict
    )

    return model

# define perform model training with parameter tunning
def perform_moodel_training_with_tuning(imputation_scenario, 
                                        model_name, 
                                        list_of_outputs, 
                                        loss_function,
                                        results_validation, 
                                        results_test,
                                        include_weigths = False, 
                                        include_dropout = False,
                                        include_reg = False, 
                                        include_lime_ph_penalty = False, 
                                        lambda_penalty = 0.001, 
                                        epochs = 100, 
                                        batch_size = 32):

    # load training data 
    x_train = pd.read_csv("../data/prepared_data/x_train_" + imputation_scenario + ".csv").values
    y_train = pd.read_csv("../data/prepared_data/y_train_" + imputation_scenario + ".csv")

    # transfrom the y values to dictionaries containing only values of predicted variables
    training_y = {key: y_train[key].values for key in list_of_outputs}
    validation_y = {key: y_val[key].values for key in list_of_outputs}

    # add dummy outputs in case the penalty term is being used (and both predictors are included, in that case it being true is a mistake)
    if "pH" in list_of_outputs and "lime" in list_of_outputs and include_lime_ph_penalty == True: 
        training_y["combined_output"] = np.zeros((len(y_train),2))
        validation_y["combined_output"] = np.zeros((len(y_val),2))

    # Initialize the tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp,  
                               list_of_outputs = list_of_outputs, 
                               x_train = x_train, 
                               loss_function = loss_function, 
                               include_dropout = include_dropout, 
                               include_lime_ph_penalty = include_lime_ph_penalty, 
                               lambda_penalty = lambda_penalty, 
                               include_reg = include_reg),
        objective = create_objectives_from_loss(loss_function, list_of_outputs, include_lime_ph_penalty), 
        max_trials=50, 
        executions_per_trial=1,
        directory="keras_tuner_logs",
        project_name = model_name 
    )

    # Define training arguments
    search_args = {
        "x": x_train,
        "y": training_y,
        "validation_data": (x_val, validation_y),
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks":[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    }

    # Add sample_weight only if include_weights is True
    if include_weigths:
        # load sample weights
        sample_weights = pd.read_csv("../data/prepared_data/sample_weights_" + imputation_scenario + ".csv")
        # restructure them into a dictionary (expected format)
        sample_weights_dict = {key: sample_weights["sample_weight_" + key].values for key in list_of_outputs}
        # add zero weights for the combined output if penalization for relationship of outputs is applied
        if "pH" in list_of_outputs and "lime" in list_of_outputs and include_lime_ph_penalty == True: 
            sample_weights_dict["combined_output"] = np.ones((len(x_train),2)) 
        # Add them to search arguments
        search_args["sample_weight"] = sample_weights_dict
    else:
        sample_weights_dict = None

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

    # Save best hyperparameters to a JSON file
    with open(f"./tuning_results/best_hyperparameters/{model_name}.json", "w") as f:
        json.dump(best_hps.values, f, indent=4)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    # Save the best model
    best_model.save("./tuning_results/best_models/"+ model_name + ".keras")

    # Evaluate best model on validation data
    metrics = best_model.evaluate(x_val, validation_y, return_dict=True)
    print("Best model metrics")
    print(metrics)
    
    # save the predicted values for vizualization
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    if len(list_of_outputs) == 1:
        output_name = list_of_outputs[0]
        results_validation[f"{model_name}.{output_name}"] = list(y_pred_val.values())[0]
        results_test[f"{model_name}.{output_name}"] = list(y_pred_test.values())[0]

    elif len(list_of_outputs) == 2:
        results_validation[model_name + ".pH"] = y_pred_val["pH"]
        results_validation[model_name + ".lime"] = y_pred_val["lime"]
        results_test[model_name + ".pH"] = y_pred_test["pH"]
        results_test[model_name + ".lime"] = y_pred_test["lime"]

    else:
        print("The output was not saved due to unexpeced number of variables")

    # save best model training history
    include_in_validation_outputs = list_of_outputs
    if "pH" in list_of_outputs and "lime" in list_of_outputs and include_lime_ph_penalty == True:
        include_in_validation_outputs.append('combined_output')

    history = best_model.fit(
    x_train,
    {output: training_y[output] for output in include_in_validation_outputs},
    validation_data=(x_val, {output: validation_y[output] for output in include_in_validation_outputs}),
    epochs= epochs,
    batch_size= batch_size,
    sample_weight=sample_weights_dict,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
        
    # Save plot of history
    plt.clf()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training History " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./tuning_results/history_plots/' + model_name + '.png')
    plt.close()

    return(results_validation, results_test)

# Example code: 
# results_validation, results_test = perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation_from_3_5_classes", 
#                                     model_name = "example",
#                                     list_of_outputs = ["lime"],
#                                     loss_function = {'lime': zero_inflated_mse_loss},
#                                     results_validation=results_validation,
#                                     results_test=results_test,
#                                     include_weigths = True,
#                                     epochs = 10
#                                     )

# results_validation.to_csv('./tuning_results/example_validation.csv', index=False)
# results_test.to_csv('./tuning_results/example_test.csv', index=False)


# Load already trainined model and perform prediction

def load_model_and_predict(model_name, inputs_prediction, df_to_save_output_to):
    loaded_model = load_model(filepath= "./tuning_results/best_models/"+ model_name + ".keras", compile = False, custom_objects={"custom_relu": custom_relu})
    # perform prediction
    predicted_values = loaded_model.predict(inputs_prediction)
    # save outputs
    if model_name.startswith("lime"):
        df_to_save_output_to[f"{model_name}.lime"] = list(predicted_values.values())[0]

    elif model_name.startswith("pH"):
        df_to_save_output_to[f"{model_name}.pH"] = list(predicted_values.values())[0]

    elif model_name.startswith("both"):
        df_to_save_output_to[model_name + ".pH"] = predicted_values["pH"]
        df_to_save_output_to[model_name + ".lime"] = predicted_values["lime"] 
    return  df_to_save_output_to




