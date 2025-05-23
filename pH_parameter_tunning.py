import pandas as pd
import numpy as np
np.random.seed(189)
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from sklearn.metrics import r2_score

# load data
# training and validation
x_train = pd.read_csv("../data/x_train.csv").values
y_train = pd.read_csv("../data/y_train.csv")
x_val = pd.read_csv("../data/x_valid.csv").values
y_val = pd.read_csv("../data/y_valid.csv")
sample_weights = pd.read_csv("../data/sample_weights.csv").values.flatten()

# test data
x_test = pd.read_csv("../data/x_test.csv").values
y_test = pd.read_csv("../data/y_test.csv")


# define dfs to save all results into
results_validation = y_val
results_test = y_test



# Define the model building function
def build_model(hp, number_outputs):
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
def perform_moodel_training_with_tuning(training_x, entire_tranining_y, model_name, weigths, list_of_dependent_var):
    training_y = entire_tranining_y[list_of_dependent_var].values
    validation_y = y_val[list_of_dependent_var].values

    # Initialize the tuner
    tuner = kt.RandomSearch(
        build_model (number_outputs = len(list_of_dependent_var)),
        objective="val_mae", # best model is selected based on lowest MAE in the validation data
        max_trials=10,
        executions_per_trial=1,
        directory="keras_tuner_logs",
        project_name = model_name 
    )

    # Search for best hyperparameters
    tuner.search(
        training_x, training_y[],
        validation_data=(x_val, validation_y),
        sample_weight=weigths,
        epochs=50,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=5)]
    )

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
    loss, mae, mse = best_model.evaluate(validation_x, validation_y)
    rmse = np.sqrt(mse)
    y_pred_val = best_model.predict(validation_x).flatten()
    r2 = r2_score(validation_y, y_pred_val)

    print(f"Metrics of best model on validation set")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # save the predicted values for vizualization
    results_validation[model_name] = y_pred_val

    y_pred_test = best_model.predict(x_test).flatten()
    results_test[model_name] = y_pred_test
    
perform_moodel_training_with_tuning(training_x = x_train, 
                                    entire_tranining_y = y_train,
                                    model_name = "pH_try_run",
                                    weigths = sample_weights,
                                    list_of_dependent_var = ["pH"])


# Export results to CSV
results_validation.to_csv('./tuning_results/predictions_validation.csv', index=False)
results_test.to_csv('./tuning_results/predictions_test.csv', index=False)


