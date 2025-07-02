import shap
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

# import the NN training script
import NN_training_script as train

#load_data
x_train = pd.read_csv("../data/prepared_data/x_train_no_lime_imputation_from_3_5_classes.csv").values
x_test = pd.read_csv("../data/prepared_data/x_test.csv").iloc[0:100, :].values
x_features = pd.read_csv("../data/prepared_data/x_test.csv").columns.values.tolist()

def create_shap_values_plot(model_name, output_variable):
    # load model
    model = load_model(filepath= "./tuning_results/best_models/"+ model_name + ".keras", compile = False, custom_objects={"custom_relu": train.custom_relu})

    def predict_array(x):
        return model.predict(x)[output_variable]

    # explain the model's predictions using SHAP
    background = shap.kmeans(x_train, 100)
    explainer = shap.KernelExplainer(predict_array,background)
    shap_values = explainer.shap_values(x_test,nsamples=100)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.summary_plot(np.squeeze(shap_values),x_test,feature_names=x_features, show=False)
    plt.savefig('../Figures/Shap_plots/' + model_name + output_variable +'.png')
    plt.cla()
    
# Univariate
create_shap_values_plot(model_name = "pH_no_lime_imputation_from_3_5_classes", output_variable = "pH")
create_shap_values_plot(model_name = "lime_no_lime_imputation_from_3_5_classes", output_variable = "lime")
# Multivariate
create_shap_values_plot(model_name = "both_no_lime_imputation_from_3_5_classes", output_variable = "pH")
create_shap_values_plot(model_name = "both_no_lime_imputation_from_3_5_classes", output_variable = "lime")
# Informed Multivariate
create_shap_values_plot(model_name = "both_no_3_5_classes_penatly_001_squared", output_variable = "pH")
create_shap_values_plot(model_name = "both_no_3_5_classes_penatly_001_squared", output_variable = "lime")