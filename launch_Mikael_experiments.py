# Copyright (C) 2023 Antonio Rodriguez
# 
# This file is part of T1DM_WARIFA.
# 
# T1DM_WARIFA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# T1DM_WARIFA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with T1DM_WARIFA.  If not, see <http://www.gnu.org/licenses/>.

import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from typing import Dict


from utils import *

from sensor_params import *

from models.training import *

# Single step forecasting models and functions 
from models.single_step.unet1D_regressor import *
from models.single_step.unet1D_regressor import get_model as get_unet1D_single_step

from models.single_step.unet1D_DIL_regressor import *
from models.single_step.unet1D_DIL_regressor import get_model as get_DIL_unet1D_single_step

from models.single_step.LSTMVanilla import *
from models.single_step.LSTMVanilla import get_model as get_LSTM_single_step

from models.single_step.unet1D_LSTM_regressor import *
from models.single_step.unet1D_LSTM_regressor import get_model as get_unet1DLSTM_single_step

from models.single_step.unet1D_nonCompres_regressor import *
from models.single_step.unet1D_nonCompres_regressor import get_model as get_unet1D_nonCompres_single_step

from models.single_step.StackedLSTM import *
from models.single_step.StackedLSTM import get_model as get_StackedLSTM_single_step

from evaluation.single_step.evaluation import model_evaluation as single_step_model_evaluation
from evaluation.single_step.evaluation import model_evaluation_refeed as single_step_model_evaluation_refeed


# Multi step forecasting models and functions
from models.multi_step.unet1D_regressor import *
from models.multi_step.unet1D_regressor import get_model as get_unet1D_multi_step

from models.multi_step.unet1D_DIL_regressor import *
from models.multi_step.unet1D_DIL_regressor import get_model as get_DIL_unet1D_multi_step

from models.multi_step.LSTMVanilla import *
from models.multi_step.LSTMVanilla import get_model as get_LSTM_multi_step

from models.multi_step.unet1D_LSTM_regressor import *
from models.multi_step.unet1D_LSTM_regressor import get_model as get_unet1DLSTM_multi_step

from models.multi_step.unet1D_nonCompres_regressor import *
from models.multi_step.unet1D_nonCompres_regressor import get_model as get_unet1D_nonCompres_multi_step

from models.multi_step.StackedLSTM import *
from models.multi_step.StackedLSTM import get_model as get_StackedLSTM_multi_step

from models.multi_step.naive_model import *
from models.multi_step.naive_model import naive_model as get_naive_multi_step

from evaluation.multi_step.evaluation import model_evaluation as multi_step_model_evaluation
from evaluation.multi_step.evaluation import model_evaluation_close_loop as multi_step_model_evaluation_refeed

 

# Dictionary with the configurations to train all models 
from training_configs import * 

# Dataset path 
DATABASE_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\Bases de datos\WARIFA\Mikael T1DM"

filename = "MIKAEL_data.json"

parent_directory = r"C:\Users\aralmeida\Downloads"
experiments_folder = r"\T1DM_pred_experiments"


def read_and_load_data()-> Dict: 
    """
    This function reads data from json, .csv, or whatever format that will depends on the sensor.
    It will also return glucose, insulin, exercise, etc. depending on the sensor. Currenty, this
    function supports the following sensors: 
        - Mikael's sensor (CGM, insulin, exercise, etc.)
    
    And the following data formats:
        - .json
    
    """
    ############## THIS WILL BE CHANGED WHEN MORE DATA IS AVAILABLE -> read_data() + preprocessed_data() function

    # Load data from the json file. If executed, comment the line and load the data from the pickle file
    # data , basal_data_dict, blood_pressure_dict, bolus_data_dict, exercise_data_dict, carbohydrates_data_dict, pump_events_data_dict, sgv_data_dict, sleep_data_dict, smgb_data_dict, steps_data_dict, weight_dict  = extract_Mikael_data(DATABASE_PATH, filename, ONLY_CGM = True)
    with open(DATABASE_PATH+'\CGM.pk1', 'rb') as f:
        sgv_data_dict = pickle.load(f)
    os.chdir(parent_directory)

    return sgv_data_dict

def launch_experiment(exp_config : Dict, kernel_size : int = 3, tau : int = 1, lr : float = 0.0001, batch_size : int = 2, epochs : int = 10) -> None:
    """
    This function launch the experiment according to the configuration specified in 'training_configs.py'.
    It includes the models, some step of the preprocessing, or the selected loss funcation, among others. 
    These parameters can be changed, but those changes must be done also in the correspondant part 
    of the code. 
    
    Launching this experiment will generate a single subdirectory for each experiment. It will contain
    training information and the most relevant results in the '/training' folder. Inside this folder,
    another one will be created, the '/evaluation' folder, that contains the main figures that allow us
    to evaluate the model performance just with a glance. 
    
    Finally, this function will generate a dictionary that  will be exported as a .json file
    (results_dictionary.json). This dictionary  will contain all the results of the experiments.
    

    Args:
    ----
        exp_config (Dict): Dictionary with the experiment configuration (check 'training_configs.py')
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        tau (int, optional): Convolution stride. Defaults to 1.
        lr (float, optional): Learning rate. Defaults to 0.0001.
        batch_size (int, optional): Batch size. Defaults to 2.
        epochs (int, optional): Number of epochs. Defaults to 10.
    """

    sgv_data_dict = read_and_load_data()

    # Test name (Dictionary with the stored configuration. Check 'training_configs.py')
    test = exp_config

    # Counter 
    i = 0
    total_exps = len(test['sensor'])*len(test['N'])*len(test['step'])*len(test['PH'])*len(test['single_multi_step'])*len(test['partition'])*len(test['normalization'])*len(test['under_over_sampling'])*len(test['model'])*len(test['loss_function'])

    # If not created, create a dictionary
    results_dictionary = create_results_dictionary(parent_directory, experiments_folder)

    # Avoid the generation of figures
    plt.ioff()

    for sensors in range(len(test['sensor'])):
        for lengths in range(len(test['N'])):
            for steps in range(len(test['step'])):
                for PHs in range(len(test['PH'])):
                    for predic_type in range(len(test['single_multi_step'])):
                        for partition in range(len(test['partition'])):
                            for norm_steps in range(len(test['normalization'])):
                                for under_over_samp in range(len(test['under_over_sampling'])):
                                    for model_names in range(len(test['model'])):
                                        for loss_function in range(len(test['loss_function'])):
                                
                                            # Update parameters 
                                            sensor = test['sensor'][sensors]
                                            N = test['N'][lengths]
                                            step = test['step'][steps]
                                            PH = test['PH'][PHs]
                                            single_multi_step = test['single_multi_step'][predic_type]
                                            data_partition = test['partition'][partition]
                                            normalization = test['normalization'][norm_steps]
                                            under_over_sampling = test['under_over_sampling'][under_over_samp]
                                            model_name = test['model'][model_names]
                                            loss_function = test['loss_function'][loss_function]

                                            key = get_dictionary_key(sensor, single_multi_step, N, step, PH, data_partition, normalization, under_over_sampling, model_name, loss_function)

                                            # If not created the directory correspondant with this configuration, create it
                                            subdirectory = r"\{}\N{}\step{}\PH{}\{}\{}\norm_{}\{}_sampling\{}\{}".format(sensor["NAME"], N, step, PH, single_multi_step,
                                                                                                                        data_partition, normalization, under_over_sampling, model_name, loss_function)
                                            if not os.path.exists(parent_directory+experiments_folder+subdirectory):
                                                os.makedirs(parent_directory+experiments_folder+subdirectory)
                                            
                                            # Go to subdirectory 
                                            os.chdir(parent_directory+experiments_folder+subdirectory)

                                            # Counter
                                            i = i+1

                                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~\nRunning experiment %d/%d:\n" % (i, total_exps))

                                            print("Configuration:\nsensor = %s\nN = %d\nstep = %d\nPH = %d\nsingle/multi step = %s\npartition = %s\nnorm = %s\nunder-over = %s\nmodel : %s\nloss funcion = %s\n" 
                                                % (sensor["NAME"], N, step, PH, single_multi_step, data_partition, normalization, under_over_sampling, model_name, loss_function))

                                            # Generate X and Y
                                            if single_multi_step == 'single':
                                                X, Y, X_times, Y_times = get_CGM_X_Y(sgv_data_dict, sensor, N, step, PH, experiments_folder, plot=False, verbose = 0)
                                            elif single_multi_step == 'multi':
                                                X, Y, X_times, Y_times = get_CGM_X_Y_multistep(sgv_data_dict, sensor, N, step, PH, experiments_folder, plot=False, verbose = 0)  
                                            else:   
                                                raise ValueError("'single' or 'multi' step forecasting must be specified in 'training_config.py'")

                                            # Data normalization
                                            if normalization == 'min-max':
                                                X_norm = (X - np.min(X))/(np.max(X) - np.min(X))
                                                Y_norm = (Y - np.min(X))/(np.max(X) - np.min(X))
                                            elif normalization == None: 
                                                X_norm = X
                                                Y_norm = Y 
                                            else: 
                                                raise ValueError("Not valid normalization: only 'min-max' or None are currently supported")
                                            
                                            # Data partition 
                                            if data_partition == 'june-21':
                
                                                # Simple partition (imitating Himar work for comparison): X_train until 30/05/2021 and X_test from 31/05/2021
                                                # Instances that include two days are removed
                                                X_train = X_norm[np.where(X_times[:,N-1] <= pd.to_datetime('2021-05-31 00:00:00'))[0]]
                                                Y_train = Y_norm[np.where((Y_times[:,0] < pd.to_datetime('2021-05-30 23:59:59')))[0]]
                                                X_test = X_norm[np.where((X_times[:,N-1] > pd.to_datetime('2021-06-01 00:00:00')))[0]]
                                                Y_test = Y_norm[np.where((Y_times[:,0] > pd.to_datetime('2021-06-01 00:00:00')))[0]] # Left non-normalized to compute the metrics

                                                print("X_train shape: ",str(X_train.shape))
                                                print("Y_train shape: ",str(Y_train.shape))
                                                print("X_test shape: ",str(X_test.shape))
                                                print("Y_test shape: ",str(Y_test.shape), "\n")

                                            elif data_partition == 'month-wise-4-folds':
                                                ##################################################### CHANGED
                                                X_train = X_norm[np.where(X_times[:,N-1] <= pd.to_datetime('2021-05-31 00:00:00'))[0]]
                                                Y_train = Y_norm[np.where((Y_times[:,0] < pd.to_datetime('2021-05-30 23:59:59')))[0]]
                                                X_test = X_norm[np.where((X_times[:,N-1] > pd.to_datetime('2021-06-01 00:00:00')))[0]]
                                                Y_test = Y_norm[np.where((Y_times[:,0] > pd.to_datetime('2021-06-01 00:00:00')))[0]] # Left non-normalized to compute the metrics
                                                pass
                                            else: 
                                                raise ValueError("Partition name not valid")

                                            # Apply (or not) undersampling or oversampling in training 
                                            if under_over_sampling == 'under':
                                                X_train, Y_train  = undersample_normal_range_outputs(X, X_train, Y_train, multi_step=False, normalization = normalization, undersampling_factor = 2)
                                            elif under_over_sampling == None: 
                                                pass
                                        
                                            # Get model instance depending on the model name and the model type (single or multi step)
                                            if single_multi_step == 'single':
                                                if model_name == '1D-UNET':
                                                    model =  get_unet1D_single_step(N=N,
                                                                    input_features = 1,
                                                                    tau=tau,
                                                                    kernel_size=kernel_size)

                                                elif model_name == '1D-UNET-non-compres':
                                                    model =  get_unet1D_nonCompres_single_step(N=N,
                                                                    input_features = 1,
                                                                    tau=tau,
                                                                    kernel_size=kernel_size)

                                                elif model_name == 'DIL-1D-UNET':
                                                    model =  get_DIL_unet1D_single_step(N=N,
                                                                    input_features = 1,
                                                                    tau=tau,
                                                                    kernel_size=kernel_size,
                                                                    dilation_rate=1)

                                                elif model_name == 'LSTM':
                                                    model =  get_LSTM_single_step(N=int(N),
                                                                    input_features = 1)

                                                elif model_name == '1D-UNET-LSTM':
                                                    model =  get_unet1DLSTM_single_step(N=N,
                                                                    input_features = 1)

                                                elif model_name == 'StackedLSTM':
                                                    model = get_StackedLSTM_single_step(N=int(N),
                                                                    input_features = 1)                                            

                                                else: 
                                                    raise ValueError("Model name not valid")
                                            
                                            elif single_multi_step == 'multi':
                                                if model_name == '1D-UNET':
                                                    model =  get_unet1D_multi_step(sensor, 
                                                                    N=N,
                                                                    input_features = 1,
                                                                    tau=tau,
                                                                    kernel_size=kernel_size)

                                                elif model_name == '1D-UNET-non-compres':
                                                    model =  get_unet1D_nonCompres_multi_step(sensor, N=N,
                                                                    input_features = 1,
                                                                    tau=tau,
                                                                    kernel_size=kernel_size,
                                                                    PH=PH)

                                                elif model_name == 'DIL-1D-UNET':
                                                    model =  get_DIL_unet1D_multi_step(N=N,
                                                                    input_features = 1,
                                                                    tau=tau,
                                                                    kernel_size=kernel_size,
                                                                    dilation_rate=1,
                                                                    PH=PH)

                                                elif model_name == 'LSTM':
                                                    model =  get_LSTM_multi_step(sensor, N=int(N),
                                                                    input_features = 1, PH=PH)

                                                elif model_name == '1D-UNET-LSTM':
                                                    model =  get_unet1DLSTM_multi_step(sensor, N=N,
                                                                    input_features = 1, PH=PH)

                                                elif model_name == 'StackedLSTM':
                                                    model = get_StackedLSTM_multi_step(sensor, N=int(N),
                                                                    input_features = 1, PH=PH)
                                                elif model_name == 'naive':
                                                    pass
                                                else: 
                                                    raise ValueError("Model name not valid")
                                                                
                                            # Number of predicting points depends on if its single step or multi-step 
                                            if single_multi_step == 'single':
                                                predicted_points = 1
                                            elif single_multi_step == 'multi':
                                                predicted_points = PH/sensor['SAMPLE_PERIOD']

                                            if model_name != 'naive':
                                                
                                                # Model training 
                                                train_model(sensor,
                                                        model,
                                                        X = X_train,
                                                        Y = Y_train,
                                                        N = N,
                                                        predicted_points = predicted_points,
                                                        epochs = epochs,
                                                        batch_size = batch_size,
                                                        lr = lr,
                                                        fold = model_name,
                                                        loss_function = loss_function,
                                                        verbose = 1 
                                                        ) 
                                            elif model_name == 'naive':
                                                print("Naive model evaluation. Training step not needed.")
                                                pass

                                            # Model evaluation depending on the forecast type: single or multi step
                                            if single_multi_step == 'single':

                                                # Non-refeed evaluation
                                                results_normal_eval = single_step_model_evaluation(N, PH, model_name, normalization, X_test, Y_test, X, loss_function)
                                                os.chdir('..')

                                                # # Refeed the model with the model output to evaluate the model
                                                # results_refeed_eval = single_step_model_evaluation_refeed(N, PH, model_name, normalization, X_test, Y_test, X)
                                                results_refeed_eval = []

                                                results_dictionary[key] = {'normal ': results_normal_eval, 'refeed': results_refeed_eval}

                                            elif single_multi_step == 'multi':

                                                # Non-refeed evaluation
                                                results_normal_eval = multi_step_model_evaluation(N, PH, model_name, normalization, X_test, Y_test, predicted_points, X, loss_function)
                                                os.chdir('..')

                                                # # Refeed the model with the FIRST model output to evaluate the model. Aiming to reduce final error. 
                                                # results_refeed_eval = multi_step_model_evaluation_refeed(N, PH, model_name, normalization, X_test, Y_test, X)
                                                results_refeed_eval = []

                                                results_dictionary[key] = {'normal ': results_normal_eval, 'refeed': results_refeed_eval}
                                        
                                            # Stop when the counter is equal to the total number of experiments
                                            if i == total_exps:
                                                break

    # Go to experiment folder
    os.chdir(parent_directory+experiments_folder)

    # Save updated dictionary 
    with open('results_dictionary.json', 'w') as fp:
            json.dump(results_dictionary, fp)