import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


from tensorflow import keras
from tensorflow.keras import layers, models

# load data
x_train = pd.read_csv("../data/x_train.csv").values
y_train = pd.read_csv("../data/y_train.csv").values
x_valid = pd.read_csv("../data/x_valid.csv").values
y_valid = pd.read_csv("../data/y_valid.csv").values

# Optional: Flatten y if it's a single column
if y_train.ndim > 1 and y_train.shape[1] == 1:
    y_train = y_train.flatten()
    y_valid = y_valid.flatten()

# 2. Build a Sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation for regression output
])

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error as an additional metric
)

# 4. Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=50,
    batch_size=32
)

# 5. Evaluate the model
loss, mae= model.evaluate(x_valid, y_valid)
print(f"Test MAE: {mae:.4f}")

# 6. Optional: Save the model
model.save("regression_model.h5")


# 1. Predict on test set
y_pred = model.predict(x_valid).flatten()  # Flatten in case it's shape (N, 1)

# 2. Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

# 3. Calculate R² score
r2 = r2_score(y_valid, y_pred)

# 4. Print results
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")