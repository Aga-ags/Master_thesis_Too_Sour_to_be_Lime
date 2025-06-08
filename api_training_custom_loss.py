import pandas as pd
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

np.random.seed(189)
import tensorflow as tf
import keras
from keras import Model, Input
from keras import ops
from keras.layers import Dense, Dropout, Concatenate
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
        # Masks when standarization is not applied: 
        # mask_zero = ops.equal(y_true, 0)
        # mask_gt_zero = ops.greater(y_true, 0)
        # Due to standarization 0 is now roughly equal to 0.5062935, which is not a nice intiger to compare to, therefore we set a tolerance level
        tolerance = 1e-4
        mask_zero = ops.abs(y_true + 0.5062935) < tolerance # values from -0.0001 to 0.0001 are considered 0
        mask_gt_zero = y_true > (-0.5062935 + tolerance) # values larger than -0.5063935 are considered larger than 0

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

def lime_ph_penalty_loss(lambda_penalty=1.0):
    def loss(y_true, y_pred):
        output_pH = y_pred[:, 0]
        output_lime = y_pred[:, 1]

        # Mask where pH < 6.5
        mask = tf.cast(tf.less(output_pH, -0.115569), tf.float32)  # 0.115568 is equvalent to pH 6.5 post standarization

        # Compute penalty
        masked_lime = output_lime * mask
        mask_sum = tf.reduce_sum(mask)
        epsilon = 1e-6
        penalty = tf.reduce_sum(tf.abs(masked_lime)) / (mask_sum + epsilon)

        return lambda_penalty * penalty
    return loss

def create_objectives_from_loss(loss_function, list_of_outputs, include_lime_ph_penalty):
    if len(list_of_outputs) == 1: 
        objectives_list = [kt.Objective(f"val_{loss}", direction="min") for key, loss in loss_function.items()]
    elif len(list_of_outputs) == 2:     
        objectives_list = [kt.Objective(f"val_{key}_{loss}", direction="min") for key, loss in loss_function.items()]
        if include_lime_ph_penalty == True: 
            objectives_list.append(kt.Objective("val_combined_output_loss", direction="min"))
    else: 
        print("Unexpected number of outputs in the objective creation function")
        objectives_list = None
    return objectives_list

# Build the neural network architecture 
def build_model(hp, list_of_outputs, x_train, loss_function, include_dropout = True, include_lime_ph_penalty = False, lambda_penalty = 0.001):
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

    # From here the model definition depends on which outputs where indicated in the list_of_outputs
    output_dict = {}
    metric_dict = {}

    if "pH" in list_of_outputs:
        output_ph = Dense(1, name="pH", activation="linear")(x) # add final layer of NN
        output_dict["pH"] = output_ph # add pH to outputs of the model
        metric_dict["pH"] = ["mae", "mse"] # add measures that make sense for pH to the metrics 

    if "lime" in list_of_outputs:
        output_lime = Dense(1, name="lime", activation=hp.Choice("activation_last_layer", values=["linear", "relu"]))(x)
        output_dict["lime"] = output_lime
        metric_dict["lime"] = ["mae", "mse", zero_inflated_mse()]

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
                                        include_weigths = True, 
                                        include_dropout = True,
                                        include_lime_ph_penalty = False, 
                                        lambda_penalty = 0.001, 
                                        epochs = 50, 
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
        lambda hp: build_model(hp,  list_of_outputs = list_of_outputs, x_train = x_train, loss_function = loss_function, include_dropout = include_dropout, include_lime_ph_penalty = include_lime_ph_penalty, lambda_penalty = lambda_penalty),
        objective = create_objectives_from_loss(loss_function, list_of_outputs, include_lime_ph_penalty), 
        max_trials=10, # 50 - change
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
        # Add them to search arguments
        sample_weights_dict = {key: sample_weights["sample_weight_" + key].values for key in list_of_outputs}
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
    best_model.save("./tuning_results/"+ model_name + ".keras")

    # Evaluate best model on validation data
    metrics = best_model.evaluate(x_val, validation_y, return_dict=True)
    print("Best model metrics")
    print(metrics)
    
    # save the predicted values for vizualization
    y_pred_val = best_model.predict(x_val)
    y_pred_test = best_model.predict(x_test)

    if len(list_of_outputs) == 1:
        output_name = list_of_outputs[0]
        results_validation[f"{model_name}.{output_name}"] = y_pred_val
        results_test[f"{model_name}.{output_name}"] = y_pred_test

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




perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "final_code_try_1",
                                    list_of_outputs = ["pH","lime"],
                                    loss_function = {'pH': 'mse', 'lime': "mse"},
                                    include_weigths = False,
                                    include_dropout = True,  
                                    include_lime_ph_penalty = True,   
                                    lambda_penalty = 0.001,                              
                                    epochs = 10, 
                                    batch_size = 32)


perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "final_code_try_2",
                                    list_of_outputs = ["pH"],
                                    loss_function = {'pH': 'mse'},
                                    include_weigths = False,
                                    include_dropout = True,  
                                    include_lime_ph_penalty = True,   
                                    lambda_penalty = 0.001,                              
                                    epochs = 10, 
                                    batch_size = 32)


perform_moodel_training_with_tuning(imputation_scenario = "no_lime_imputation", 
                                    model_name = "final_code_try_3",
                                    list_of_outputs = ["lime"],
                                    loss_function = {'lime': "mse"},
                                    include_weigths = False,
                                    include_dropout = True,  
                                    include_lime_ph_penalty = False,              
                                    epochs = 10, 
                                    batch_size = 32)



results_validation.to_csv('./tuning_results/functional_api_val.csv', index=False)
results_test.to_csv('./tuning_results/functional_api_test.csv', index=False)





