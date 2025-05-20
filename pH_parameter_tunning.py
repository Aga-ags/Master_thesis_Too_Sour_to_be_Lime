import pandas as pd
import numpy as np
np.random.seed(156)
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from sklearn.metrics import r2_score


# load data
x_train = pd.read_csv("../data/x_train.csv").values
y_train = pd.read_csv("../data/y_train.csv").values.flatten()
x_val = pd.read_csv("../data/x_valid.csv").values
y_val = pd.read_csv("../data/y_valid.csv").values.flatten()
sample_weights = pd.read_csv("../data/sample_weights.csv").values.flatten()

# --- Define the model building function ---
def build_model(hp):
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
    
    # Output layer (for regression)
    model.add(layers.Dense(1))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="mse",
        metrics=["mae", "mse"]
    )
    
    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=10,
    executions_per_trial=1,
    directory="keras_tuner_logs",
    project_name="regression_tuning"
)

# Search for best hyperparameters
tuner.search(
    x_train, y_train,
    validation_data=(x_val, y_val),
    sample_weight=sample_weights,
    epochs=50,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)]
)

# print search summary
tuner.search_space_summary()

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Print best hyperparameters
print("Best Hyperparameters:")
for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")

# Evaluate best model
loss, mae, mse = best_model.evaluate(x_val, y_val)
rmse = np.sqrt(mse)
y_pred = best_model.predict(x_val).flatten()
r2 = r2_score(y_val, y_pred)

print(f"Metrics of best model on validation set")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# save the best model
best_model.save("./tuning_results/best_model_pH.h5")