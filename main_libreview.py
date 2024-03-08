import os 
import numpy as np 
import pandas as pd
import sys
from typing import Dict 
import matplotlib.pyplot as plt 
import openpyxl
import pickle
import json 
from keras import backend as K

sys.path.append("..")

from libreview_utils import *
from models.training import *
from sensor_params import *
from utils import get_LibreView_CGM_X_Y_multistep, undersample_normal_range_outputs, generate_ranges_tags, generate_weights_vector
from training_configs import *

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

# Dataset path 
DATASET_PATH = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims"

def launch_LibreView_experiments(test : Dict, input_features : int, weighted_samples : bool,  kernel_size : int = 3, tau : int = 1, lr : float = 0.0001,
                                batch_size : int = 2, epochs : int = 10, dataset_path : str = DATASET_PATH) -> None:
    """
    This function launches the experiments according to the configuration specified in 'training_configs.py'.
    It includes the LibreView raw data reading and preparation, the models, some step of the preprocessing,
    or the selected loss function, among others.  These parameters can be changed, but those changes must
    be done also in the correspondant part of the code. 
    
    Launching this experiment will generate a single subdirectory for each experiment. It will contain
    training information and the most relevant results in the '/training' folder. Inside the '/evaluation' folder,
    the main figures that allow us to evaluate the model performance just with a glance..  
    
    Besides, one folder per patient is generated. Finally, this function will generate a dictionary per patient that 
    will be exported as a .json file (results_dictionary.json). This dictionary  will contain all the results of
    the experiments.
    
    Args:
    ----
        test (Dict): Dictionary with the experiment configuration (check 'training_configs.py')
        input_features (int): Number of input features. Currently, 1 is the CGM, and 2 is the CGM and its derivative.
        weighted_samples (bool): If True, the samples are weighted according to the class distribution and its importance (weights obtained after heuristically)
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        tau (int, optional): Convolution stride. Defaults to 1.
        lr (float, optional): Learning rate. Defaults to 0.0001.
        batch_size (int, optional): Batch size. Defaults to 2
        epochs (int, optional): Number of epochs. Defaults to 10.
        dataset_path (str, optional): Path to the LibreView raw data stored in .csv. Defaults to DATASET_PATH.
    
    Returns:
    -------
        None    
    """

    # Initial time
    t0 = time.time()
    
    # From the .csv files. extract the oldest year and store them to load them separately 
    get_oldest_year_npys_from_LibreView_csv(dataset_path)

    total_exps = len(test['sensor'])*len(test['N'])*len(test['step'])*len(test['PH'])*len(test['single_multi_step'])*len(test['partition'])*len(test['normalization'])*len(test['under_over_sampling'])*len(test['model'])*len(test['loss_function'])

    plt.ioff()
    
    # Iterate over the ID folders to generate the 4-folds 
    for id in os.listdir(): 

        # Counter 
        i = 0

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id): 
        # if ('045' in id) and ('npy' not in id) and ('txt' not in id) : 
        
            # Get into the ID patient folder
            os.chdir(id)

            # Create dictionary to fill it with the results (one per patient) 
            results_dictionary = create_LibreView_results_dictionary()

            # Avoid the generation of figures
            plt.ioff()
        
            # Only read the OLDEST year of recording
            recordings = np.load('oldest_1yr_CGM.npy')
            timestamps = np.load('oldest_1yr_CGM_timestamp.npy', allow_pickle=True)

            # For each patient (ID), the experiments included in "training_configs.py" are executed
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

                                                    # Create the key for current experiment 
                                                    key = '{}_N{}_step{}_PH{}_{}_{}_{}_{}_{}'.format(single_multi_step, N, step, PH, data_partition, normalization,
                                                        under_over_sampling, model_name, loss_function)
                                                    
                                                    # Declare the dictionary key empty to fill it later if it has not been created yet 
                                                    if key not in results_dictionary.keys():
                                                        results_dictionary[key] = {}
                                                    else: 
                                                        pass

                                                    if model_name not in results_dictionary[key].keys():
                                                        results_dictionary[key][model_name] = {}
                                                    else:
                                                        pass

                                                    # Store current working directory 
                                                    cwd = os.getcwd()
                                                    
                                                    # If not created the directory correspondant with this configuration, create it
                                                    subdirectory = r"\N{}\step{}\PH{}\{}\{}\norm_{}\{}_sampling\{}\{}".format(N, step, PH, single_multi_step,
                                                                                                                            data_partition, normalization, under_over_sampling, model_name, loss_function)
                                                    if not os.path.exists(cwd+subdirectory):
                                                        os.makedirs(cwd+subdirectory)
                                                
                                                    # Go that subdirectory 
                                                    os.chdir(cwd+subdirectory)

                                                    # Counter (with printing purposes)
                                                    i = i+1

                                                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~\nRunning experiment %d/%d for Patient #%s :\n" % (i, total_exps, id))

                                                    print("Configuration:\nsensor = %s\nN = %d\nstep = %d\nPH = %d\nsingle/multi step = %s\npartition = %s\nnorm = %s\nunder-over = %s\nmodel : %s\nloss funcion = %s\n" 
                                                    % (sensor["NAME"], N, step, PH, single_multi_step, data_partition, normalization, under_over_sampling, model_name, loss_function))

                                                    # Generate X and Y (only multistep supportd)
                                                    if single_multi_step == 'multi':
                                                        X, Y, X_times, Y_times = get_LibreView_CGM_X_Y_multistep(recordings, timestamps, libreview_sensors, 
                                                                N, step, PH, plot = True, verbose = 0) 
                                                    else:   
                                                        raise ValueError("Only 'multi' step prediction is supported in the LibreView-extracted data")
                                                    
                                                    # Generate the tags associated to each Y vector ("hyper", "hypo", "normal") depending of it it contains hyper, hypo or normal values
                                                    levels_tags = generate_ranges_tags(Y)
                                                    
                                                    # Data normalization
                                                    if normalization == 'min-max':
                                                        X_norm = (X - np.min(X))/(np.max(X) - np.min(X))
                                                        Y_norm = (Y - np.min(X))/(np.max(X) - np.min(X))
                                                    elif normalization == None: 
                                                        X_norm = X
                                                        Y_norm = Y 
                                                    else: 
                                                        raise ValueError("Not valid normalization: only 'min-max' or None are currently supported")
                                                    
                                                    # If the models are multi-input, the CGM derivative must be generated
                                                    if input_features == 1: 
                                                        pass
                                                    
                                                    elif input_features == 2:

                                                        # Get 1st derivative of X_norm
                                                        X_norm_der = np.diff(X_norm, axis = 1)

                                                        # Add the last point of X_norm_dev on the right of the array tp have same dimension than X_norm
                                                        X_norm_der = np.insert(X_norm_der, -1, X_norm_der[:,-1], axis = 1)

                                                        # Stack X_norm and X_norm_der
                                                        X_norm = np.dstack((X_norm, X_norm_der))

                                                    else: 
                                                        raise ValueError("Not valid number of input features: only 1 or 2 (CGM and its derivative) are currently supported")
                                                
                                                    # Data partition (depending on if it is single input or multi-input)
                                                    if input_features == 1: 
                                                    
                                                        # Month-wise 4-folds
                                                        if data_partition == 'month-wise-4-folds':                                                 
                                                            training_cv_folds  = month_wise_LibreView_4fold_cv(X_norm, Y_norm, X_times, Y_times, N)
                                                        else: 
                                                            raise ValueError("Partition name not valid: currently 'month-wise-4-folds' is supported")
                                                    
                                                    if input_features == 2:
                                                        
                                                        # Month-wise 4-folds
                                                        if data_partition == 'month-wise-4-folds':                                                 
                                                            training_cv_folds  = month_wise_multi_input_LibreView_4fold_cv(X_norm, Y_norm, X_times, Y_times, levels_tags, N, input_features)
                                                        else: 
                                                            raise ValueError("Partition name not valid: currently 'month-wise-4-folds' is supported")

                                                    # Apply (or not) undersampling or oversampling in training 
                                                    if under_over_sampling == 'under':
                                                        X_train, Y_train  = undersample_normal_range_outputs(X, X_train, Y_train, multi_step=False, normalization = normalization, undersampling_factor = 2)
                                                    elif under_over_sampling == None: 
                                                        pass

                                                    if single_multi_step == 'multi':
                                                        if model_name == '1D-UNET':
                                                            model =  get_unet1D_multi_step(sensor, 
                                                                            N=N,
                                                                            input_features = input_features,
                                                                            tau=tau,
                                                                            kernel_size=kernel_size, 
                                                                            PH=PH)
                                                            model.save_weights('initial_weights.h5')                                                        
                                                        
                                                        elif model_name == '1D-UNET-non-compres':
                                                            model =  get_unet1D_nonCompres_multi_step(sensor, N=N,
                                                                            input_features = input_features,
                                                                            tau=tau,
                                                                            kernel_size=kernel_size,
                                                                            PH=PH)
                                                            model.save_weights('initial_weights.h5')

                                                        elif model_name == 'DIL-1D-UNET':
                                                            model =  get_DIL_unet1D_multi_step(sensor, 
                                                                            N=N,
                                                                            input_features = input_features,
                                                                            tau=tau,
                                                                            kernel_size=kernel_size,
                                                                            dilation_rate=1,
                                                                            PH=PH)
                                                            model.save_weights('initial_weights.h5')

                                                        elif model_name == 'LSTM':
                                                            model =  get_LSTM_multi_step(sensor, N=int(N),
                                                                            input_features = input_features, PH=PH)
                                                            model.save_weights('initial_weights.h5')

                                                        elif model_name == '1D-UNET-LSTM':
                                                            model =  get_unet1DLSTM_multi_step(sensor, N=N,
                                                                            input_features = input_features, PH=PH)
                                                            model.save_weights('initial_weights.h5')

                                                        elif model_name == 'StackedLSTM':
                                                            model = get_StackedLSTM_multi_step(sensor, N=int(N),
                                                                            input_features = input_features, PH=PH)
                                                            model.save_weights('initial_weights.h5')
                                                        elif model_name == 'naive':
                                                            pass
                                                        else: 
                                                            raise ValueError("Model name not valid")
                                                    else: 
                                                        raise ValueError("Only 'multi' step prediction is supported in the LibreView-extracted data")
                                                    
                                                    # Compute the number of predicted points that depends on the PH and the sensor sampling period
                                                    predicted_points = round(PH/sensor['SAMPLE_PERIOD'])

                                                    # If model is "naive", there is nothing to train
                                                    if model_name != 'naive':
                                                        
                                                        if data_partition == 'month-wise-4-folds':

                                                            # Train and evaluate each folds separately 
                                                            for fold in training_cv_folds.keys():

                                                                # The model is reinitialized for each fold
                                                                model.load_weights('initial_weights.h5')                                                                                                                            
                                                                
                                                                # If the directory fold is not created, create it
                                                                if fold not in os.listdir():
                                                                    os.mkdir(fold)
                                                                
                                                                # Get into the fold directory
                                                                os.chdir(fold)

                                                                # WEIGHTS COMPUTATION: Count the number of samples in each category
                                                                if weighted_samples:
                                                                    weights = generate_weights_vector(training_cv_folds[fold]['train_tags'])
                                                                else:
                                                                    weights = []
                                                                
                                                                # One model training per fold
                                                                train_model(sensor,
                                                                            model,
                                                                            X = training_cv_folds[fold]['X_train'],
                                                                            Y = training_cv_folds[fold]['Y_train'],
                                                                            N = N,
                                                                            predicted_points = predicted_points,
                                                                            epochs = epochs,
                                                                            batch_size = batch_size,
                                                                            lr = lr,
                                                                            fold = id+"-"+model_name+"-"+fold,
                                                                            sample_weights=weights, 
                                                                            loss_function = loss_function,
                                                                            verbose = 1 
                                                                            )
                                                            
                                                                # Model evaluation 
                                                                results_normal_eval = multi_step_model_evaluation(N, PH, id+"-"+model_name+"-"+fold, normalization, input_features, training_cv_folds[fold]['X_test'],
                                                                                                                training_cv_folds[fold]['Y_test'], predicted_points, X, loss_function, plot_results=False)
                                                                
                                                                # # Refeed the model with the FIRST model output to evaluate the model. Aiming to reduce final error. 
                                                                # results_refeed_eval = multi_step_model_evaluation_refeed(N, PH, model_name, normalization, X_test, Y_test, X)
                                                                results_refeed_eval = []

                                                                results_dictionary[key][model_name][fold] = {'normal ': results_normal_eval, 'refeed': results_refeed_eval}

                                                                # Back to fold directory 
                                                                os.chdir('../..')

                                                        else:
                                                            raise ValueError("Partition name not valid: currently only 'month-wise-4-folds' training is supported")

                                                        # Back to id folder
                                                        os.chdir(cwd)

                                                        # Save updated dictionary 
                                                        with open('results_dictionary.json', 'w') as fp:
                                                            json.dump(results_dictionary, fp)                                                      

                                                        # # Back to previous directory 
                                                        # os.chdir('..')

                                                    elif model_name == 'naive':
                                                        print("Naive model evaluation. Training step not needed. Only evaluation is performed")

                                                        # Do the naive prediction in all folds 
                                                        for fold in training_cv_folds.keys():
                                                            
                                                                # If the directory fold is not created, create it
                                                                if fold not in os.listdir():
                                                                    os.mkdir(fold)
                                                                
                                                                # Get into the fold directory
                                                                os.chdir(fold)

                                                                if single_multi_step == 'multi':

                                                                    # Non-refeed evaluation
                                                                    results_normal_eval = multi_step_model_evaluation(N, PH, id+"-"+model_name+"-"+fold, normalization, input_features, 
                                                                                                                    training_cv_folds[fold]['X_test'], training_cv_folds[fold]['Y_test'], predicted_points, X, loss_function,
                                                                                                                    plot_results=False)
                                                                    # os.chdir('..')

                                                                    # # Refeed the model with the FIRST model output to evaluate the model. Aiming to reduce final error. 
                                                                    # results_refeed_eval = multi_step_model_evaluation_refeed(N, PH, model_name, normalization, X_test, Y_test, X)
                                                                    results_refeed_eval = []

                                                                    results_dictionary[key][model_name][fold] = {'normal ': results_normal_eval, 'refeed': results_refeed_eval}
                                                                    
                                                                    # Back to previous directory 
                                                                    os.chdir('../..')

                                                                    # Clear session to avoid that models stop training earlier than desired
                                                                    K.clear_session()
                                                        
                                                                else: 
                                                                    raise ValueError("Only 'multi' step prediction is supported in the LibreView-extracted data")
                                                                
                                                        
                                                        # Back to id folder
                                                        os.chdir(cwd)

                                                        # Save updated dictionary 
                                                        with open('results_dictionary.json', 'w') as fp:
                                                            json.dump(results_dictionary, fp)                                                      

                                                        # # Back to previous directory 
                                                        # os.chdir('..')
                                                    
                                                    else: 
                                                        raise ValueError("Model name not valid")

                                                    # Close all figures 
                                                    plt.close('all')

            # AQUI ES DONDE ACABA EL BUCLE Y HAY QUE PASAR AL SIGUIENTE ID
            # Back to previous directory 
            os.chdir('..')

            # Stop when the counter is equal to the total number of experiments
            if i == total_exps*29: # 29 is the number of patients
                break
    
    # Final time
    t1 = time.time()
    print("Time elapsed: %d seconds" % (t1-t0))