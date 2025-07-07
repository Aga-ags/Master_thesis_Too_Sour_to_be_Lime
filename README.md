This repository contains scripts used to process, train Neural Network models and analyze data for Master's thesis "Too Sour to be Lime: Improving Consistency of Digital Soil Mapping with multivariate neural network and Soil Science Informed Loss Constraint" by Agnieszka Kubica.
The code can be used to perform Digital Soil Mapping of pH and lime content of arable land of Zurich. Data is available upon request.

The scripts include (in suggested ordered of execution):
Data processing:
1. data_processing.Rmd - data processing of data for training and testing models
2. arable_land_prediction_df.Rmd - data processing of Canton of Zurich rasters for performing prediction on the enitrety of arable land

Neural network training:
1. NN_training_script.py - script containing all functions needed to train the Neural Networks
2. imputation_scenario_pH.py - training univatiate pH models under different imputation scenarios
3. imputation_scenario_lime.py - training univatiate lime content models under different imputation scenarios
4. imputation_scenario_both.py - training multivariate lime content models under different imputation scenarios
5. overfitting_tuning.py - training models with various techniques to prevent overfitting
6. lambda_tuning_linear.py - training multivariate model with linear penalty term with different weights
7. lambda_tunning_quadratic.py - training multivariate model with quadratic penalty term with different weights

Prediction:
1. prediction_final_models.py - predicting with final models for the test set and entirety of arable land 

Model evaluation: 
1. prediction_evaluation.Rmd - evaluating the accuracy and consitency of prediciton
2. shap_values_final_models.py - calculating feature importance (shap values) for the final models

