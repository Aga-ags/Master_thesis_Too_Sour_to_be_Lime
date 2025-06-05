from keras.models import load_model
import pickle
import matplotlib.pyplot as plt

models_list = ["both_imp_and_lime_65_0_2_try2", "lime_imp_and_lime_65_try2", "lime_imp_and_lime_65_0_2_try2", "lime_imp_and_lime_65_0_2_l1", "lime_imp_and_lime_65_0_2_l2"]

for model_name in models_list:
    # Load model
    model = load_model("./tuning_results/" + model_name +".keras")

    # Load training history
    with open("./tuning_results/" + model_name + "_history.pkl", "rb") as f:
        history = pickle.load(f)

    # Plot history
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training History " + model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()