# Copyright (C) 2024 Antonio J. Rodriguez-Almeida
# 
# This file is part of _______.
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
    
"""
This function encapsulates the Do It Yourself (DIY) module for glucose prediction.
After experimentation (see main_libreview.py), this function can be considered 
the DIY module itself ready to be used. The input of the module is a .cvs (or .json) file.
The output of this module is: 
    a) If it is the user's first use, it runs the normal training of the DL model following a
    4-folds month-wise CV approach. The model that presents better performance is saved as a .h5 file.
    After this, T1D-related personal data analysis an visualization is provided, and
    an 1-hour prediction is performed and depicted using the last 24 hours of the user's data (96 data points
    at 15-minute intervals for the LibreView data used to develop this framework). 
    ***** THE CHOICE OF THE BEST MODEL MIGHT CHANGE IN SUBSEQUENT VERSIONS OF THIS FRAMEWORK *****
    b) If the user has already a DL model, (i.e., step a) has been done once) the function loads the model,
    performs the same analysis with the new updated user's data, and performs a new 1-hour prediction taking 
    the last 24 hours of the user's data.

Args: 
----
    user_file : user's file. This file should be a .csv or .json file with the user's data. ***More formats to come

Returns:
-------
    results : a dictionary to visualize the results even no new data is provided by the user, but whenever the user wishes so. 

"""

import os 
import tensorflow as tf
import sys
import time
sys.path.append("..")
from app.your_data_read_and_analysis import *
from models.training import month_wise_multi_input_LibreView_4fold_cv, train_model, ISO_adapted_loss
from models.multi_step.LSTMVanilla import get_model as get_LSTM_multi_step
from evaluation.multi_step.evaluation import model_evaluation as multi_step_model_evaluation
from utils import get_LibreView_CGM_X_Y_multistep, generate_ranges_tags, generate_weights_vector
from app.app_visualization import *
from typing import Tuple
import warnings
from your_AI_DIY_parameters import *

import matplotlib.pyplot as plt 

from sensor_params import *

# # Go to the directory where the .csv is stored (TESTING PURPOSES)
# os.chdir(r"C:\Users\aralmeida\Downloads\prueba_app_CGM")
# # os.chdir(r"C:\Users\aralmeida\Downloads\prueba_app_CGM_ID_no_valido")
# user_file = "ID001_S001_R001_glucose_12-6-2023.csv" # Testing, integration could be different 
# # user_file = "ID066_S001_R001_glucose_6-9-2023.csv"

# Ignore warnings
warnings.filterwarnings("ignore")

# Avoid TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# WELCOME TO THE AI BASED DIY YOURSELF TOOL, UPLOAD YOUR DATA AND GET YOUR RESULTS
# These welcome messages are shown just the first time the module is used (considering first time if the model hasn't been generated)
if "Your_AI_CGM_predictor" not in os.listdir():
    print("Welcome to the AI-based Do-It-Yourself glucose predictor!\n\n")
    print("Here, you will be able to have an in-depth analysis of your uploaded CGM data, as well as an AI-based prediction of your glucose level in the next hour!")
    print("\nThis module has been validated, but it is still work in progress. Currently, the following sensor models are supported:\n")
    print("\t- Freestyle Libre 2\n\t- FreeStyle LibreLink\n\t- LibreLink")

    print("\nThe following input data is required (and supported) for this module:\n\t- CGM data (for now, at 15 minutes intervals)")

    print("\nWe are currently working on:\n\t- Support for more sensor models\n\t- Adding more data, such as insuline, physical activity, etc.\n\t- More accurate predictions\n\t- More in-depth data analysis\n\t- More user-friendly interface\n\t- More user-friendly error messages\n\t- More user-friendly documentation\n\t- More user-friendly everything")

    print("\n\n***IMPORTANT***: due to regulatory issues, this module (unfortunately) don't access to the sensor data in real time, so every time that you want to have a 1-hour prediction, you need to upload your CGM data.")
    print("***DISCLAIMER***: Please, do not use this tool as a replacement for professional medical advice, but always as a complementary tool to help you manage your T1D")

    print("\nPlease, for any suggestions, feedback or bug reports, contact the developer at: antorguez95@hotmail.com\nYou can also visit the project's GitHub page at:")
    print("https://github.com/antorguez95/Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool")

    # First of all, the users is asked about the directory where the data is stored
    print("Let's begin!\n1) If you haven't download your CGM data, please do it!\n")
    print("2) Once you have your data, please, type the path where your data is stored (e.g., C:/Users/your_user/Downloads/your_data_folder):")
    your_data_path = input()


    # Then, since the preprocesisng and the AI architecture depends on the sensor, the user is asked about the sensor he or she is using 
    print("\nThank you! Now, please, type the letter corresponding to the glucose sensor model you are using:\n")
    print("a) Abbott - Freestyle Libre 2\nb) Abbott - FreeStyle LibreLink\nc) Abbott - LibreLink\n")

    print("\n\n***ONLY Abbott sensors have been validated in this version of the tool.***") 

    ans = input()

    if ans == "a" or ans == "b" or ans == "c":
        sensor = libreview_sensors
        sensor_name = "Freestyle Libre 2"
    else:
        print("\nSorry, but this tool does not support your sensor. We are currently working on the inclusion of more sensors.")

    print("Nice! Could you confirm that your glucose sensor is a " + sensor_name+ "? Just to double check! (y/n)")
    ans = input()

    if ans == "y":
        print("\nGreat! Let's move on!")
    else:
        print("\nSorry, but this tool does not support your sensor. We are currently working on the inclusion of more sensors.")

    # Now, the user is asked about the unit of the glucose data
    print("\nPlease, type 'a' if the unit of your glucose data is in 'mg/d' or, 'b' if it is in 'mmol/L' instead\n")
    ans = input()

    if ans == "a":
        unit = "mg/dL"
    elif ans == "b":
        unit = "mmol/L"
    else:
        print("\nSorry, but you introduced a wrong letter. Please, type only 'a' or 'b'")

    print("\nSo, your glucose data is in " + unit + ", are we right? (y/n)")
    ans = input()
    if ans == "y":
        print("\nGreat! Let's move on!")
    else:
        print("\nOh, Sorry! Move back to the previous step and make sure you introduce the right unit! We'll be waiting for you!")
        exit()
    
    # your_data_path = os.getcwd()

    # Save unit for further uses
    np.save('unit.npy', unit)

    # Set Keras to float 64 to work with the ISO-loss function
    tf.keras.backend.set_floatx('float64')

    if '.h5' not in os.listdir():
        print("Oh! It seems that you don't have your personalized-AI glucose predictor yet.\nNow, your data will be analyzed.")
        print("If your data does not contain many interruptions, and the CGM samples are enough, your personalized-AI glucose predictor will be generated using your recently uploaded data.") 

        # Analyze the data
        print("\n\nAnalyzing your data... This could take a few seconds.\n\n")

        data_suitability = get_your_oldest_year_npys_from_LibreView_csv(your_data_path)

        # If data is suitable for AI:
        if data_suitability:
            print("Congrats! The data you provided is enough to generate and train your personalized-AI glucose predictor!\n")

            print("Before proceeding to the AI-model generation, would you like to know more about how your data will be used to generate your personalized-AI glucose predictor? (y/n):")
            ans = input()

            if ans == "y":

                print("\n\nSHIT ABOUT YOUR DATA\n\n")
                ###### DATA ANALYSIS ######
                ##########################
                ##########################
                ##########################
                ##########################

                print("Please, type 'next' once you have finished looking around your data to move to your AI glucose predictor")
                move_on = input()

                if move_on == "next": # A loop here would be nice to not go forward until the user type 'next'
                    print("\nNice! Now, please, be patient, since this process may take between 1 and 2 hours to complete.")
                    print("You will be notified when the process is completed.")
                else: 
                    raise Exception("Please, enter 'next' to continue.")
                            
            elif ans == "n":

                print("\nNice! Now, please, be patient, since this process may take between 1 and 2 hours to complete.")
                print("You will be notified when the process is completed.")
        
            # Raise exception
            else: 
                raise Exception("Please, enter a 'y' or 'n' to continue.")
            
            ######## CODE OF THE IMPLEMENTED AI FRAMEWORK ########

            # TO CHANGE THIS, ADD A DICTIONARY IN your_AI_DIY_parameters.py, or change the parameters of the current dictionary
            N = first_DIY_version['N'] # 96
            step = first_DIY_version['step'] # 1
            PH = first_DIY_version['PH'] # 30
            input_features = first_DIY_version['input_features'] # 2
            normalization = first_DIY_version['normalization'] # 'min-max'
            loss_function = first_DIY_version['loss_function'] # 'ISO_loss'

            epochs = 1 # real value from paper: 20
            batch_size = 512 # real value from paper: 1
            lr = 0.0001
            ##################################################

            # Load user's data and the associated timestamps 
            recordings = np.load('oldest_1yr_CGM.npy')
            timestamps = np.load('oldest_1yr_CGM_timestamp.npy', allow_pickle=True)

            # If the directory "Your_AI_CGM_predictor" is not created, create it
            if "Your_AI_CGM_predictor" not in os.listdir():
                    os.mkdir("Your_AI_CGM_predictor")
                
            # Get into the fold directory
            os.chdir("Your_AI_CGM_predictor")

            # Generating the X and Y to train the AI model. 
            X, Y, X_times, Y_times = get_LibreView_CGM_X_Y_multistep(recordings, timestamps, libreview_sensors, N, step, PH, plot = True, verbose = 0) 

            # Generate the tags associated to each Y vector ("hyper", "hypo", "normal") depending of it it contains hyper, hypo or normal values
            levels_tags = generate_ranges_tags(Y)

            # Min-max normalization 
            X_norm = (X - np.min(X))/(np.max(X) - np.min(X))
            Y_norm = (Y - np.min(X))/(np.max(X) - np.min(X))

            # Get 1st derivative of X_norm and concatenated to the original vector
            X_norm_der = np.diff(X_norm, axis = 1)
            X_norm_der = np.insert(X_norm_der, -1, X_norm_der[:,-1], axis = 1)
            X_norm = np.dstack((X_norm, X_norm_der))

            # Generate model (LSTM until now until results are generated)
            model =  get_LSTM_multi_step(sensor, N=int(N), input_features = input_features, PH=PH)
            model.save_weights('initial_weights.h5')

            # Compute the number of predicted points that depends on the PH and the sensor sampling period
            predicted_points = round(PH/sensor['SAMPLE_PERIOD'])

            # Generate the 4-folds # ASSUME THAT THE CHOICE PARTITION IS 4-FOLDS. Changes here imply changes in more parts of the code
            training_cv_folds  = month_wise_multi_input_LibreView_4fold_cv(X_norm, Y_norm, X_times, Y_times, levels_tags, N, input_features)

            # Results dictionary since the results plotting will depend on RMSE and the model choice on Parkes AB
            results_dictionary = {}

            # Train and evaluate each folds separately 
            for fold in training_cv_folds.keys():

                # The model is reinitialized for each fold
                model.load_weights('initial_weights.h5') 

                # If the directory fold is not created, create it
                if fold not in os.listdir():
                    os.mkdir(fold)
                
                # Get into the fold directory
                os.chdir(fold)

                # Models are trained with weighted samples 
                weights = generate_weights_vector(training_cv_folds[fold]['train_tags']) 

                # Initial time
                t0 = time.time()

                # One model training per fold
                print("\n\nTraining your personalized-AI glucose predictor with your data...")
                
                train_model(sensor,
                            model,
                            X = training_cv_folds[fold]['X_train'],
                            Y = training_cv_folds[fold]['Y_train'],
                            N = N,
                            predicted_points = predicted_points,
                            epochs = epochs,
                            batch_size = batch_size,
                            lr = lr,
                            fold = fold,
                            sample_weights=weights, 
                            loss_function = loss_function,
                            verbose = 0 
                            )
                
                # Final time
                t1 = time.time()

                # Model evaluation: SHOW TO USERS? DON'T THINK SO. JUST FOR DEVELOPERS AND MODEL MAINTAINERS
                results_normal_eval = multi_step_model_evaluation(N, PH, fold, normalization, input_features, training_cv_folds[fold]['X_test'],
                                        training_cv_folds[fold]['Y_test'], predicted_points, X, loss_function, plot_results=False) 

                results_dictionary[fold] = results_normal_eval

                # Back to fold directory 
                os.chdir('../..')

            ######## END OF THE CODE RELATED TO THE AI FRAMEWORK ########

            # Choice of the best model considering Parkes AB percentage of the las sample of the prediction
            # Initialized to 0, will stored the current best parkes in the loop
            curr_best_parkes = 0 
            curr_parkes = 0
            fold_idx = 0 # contain the current fold
            best_fold_idx = 0 # contain the current best fold

            # Loop over the metrics (idx = 1 because it considers only the 30' prediction)
            for fold in results_dictionary.keys(): 

                # First iteration is different
                if fold_idx == 0: 
                    fold_idx = 1
                    best_fold_idx = 1
                    curr_parkes = results_dictionary['1-fold']['PARKES'][predicted_points-1]
                    curr_best_parkes = results_dictionary['1-fold']['PARKES'][predicted_points-1]
                
                else: 
                    # Update with current fold idx and correspondant Parkes 
                    fold_idx = fold_idx+1
                    curr_parkes = results_dictionary[fold]['PARKES'][predicted_points-1]

                    # Comparison and update if current Parkes is better
                    if curr_parkes > curr_best_parkes: 
                        curr_best_parkes = curr_parkes 
                        best_fold_idx = fold_idx
                    else: 
                        pass

            match best_fold_idx: 
                case 1 : 
                    best_model_key = '1-fold'
                case 2: 
                    best_model_key = '2-fold'
                case 3 : 
                    best_model_key = '3-fold'
                case 4: 
                    best_model_key = '4-fold'

            print("Congrats! Your personalized-AI model for CGM prediction has been successfully generated!\n")
            print("Time ellapsed to do it: ", t1-t0, " seconds.")
            
            # Extract the rmse to plot the prediction with the error bar associated to it
            rmse = results_dictionary[fold]['RMSE']

            # Make a prediction using the last day (i.e., 96 samples) of the data to show it to the user
            print("Making a CGM prediction of your next 30 minutes using the last day oy the data you just provided...")

            # Go to the directory where the best model is placed (depends on previous evaluation)
            os.chdir(best_model_key)
            os.chdir("training")

            # Load the model
            model = get_LSTM_multi_step(sensor, N=int(N), input_features = input_features, PH=PH) # Assumes LSTM
            model.load_weights(best_model_key+'.h5')

            # Go back to "Your_AI_CGM_predictor" directory, where the model will be called further times and save model
            os.chdir("../..")
            model.save("your_AI_based_CGM_predictor.h5")

            # Load the last day of the data
            last_day_norm = X_norm[-96:,-1,:]

            # Reshape to None, 96, 2
            last_day_norm = last_day_norm.reshape(1, 96, 2)

            # Make the prediction
            prediction = model.predict(last_day_norm)

            # Extract only the original CGM
            last_day_norm = last_day_norm[:,:,0]

            # Reshape to ,96
            last_day_norm = last_day_norm.reshape(96)

            # Denormalize 
            last_day = last_day_norm*(np.max(X) - np.min(X)) + np.min(X)

            # Take only the last instance of X_times
            last_day_timestamps = X_times[-1]
            
            # Plot and save the prediction graphics 
            get_prediction_graphic(last_day, last_day_norm, predicted_points, last_day_timestamps, rmse, unit, prediction)

            # Save the fold and the rmse to know it for further iterations 
            np.save('best_fold.npy', best_fold_idx)
            np.save('rmse.npy', rmse)

        # If data is not suitable for AI:
        if not data_suitability:

            print("Sorry, but the data you provided is not enough to generate and train your personalized-AI glucose predictor.")
            print("Please, try again with more data samples.")
            print("If you think that this is an error, please contact the developer/maintainer at: _____")

            print("\n\nDo you want to know more about why did your data not meet the criteria to generate your personalized-AI glucose predictor? (y/n): ")
            ans = input()

            if ans == "y": 
                
                #################################################
                # RELLENAR CON INFO SOBRE POR QUE NO SE PUEDE
                #################################################
                #################################################
                #################################################
                # Information to EMPOWER the user and make him/her understand the process and the requirements to generate the personalized-AI glucose predictor
                print("\n\n\nDETALLES DE QUE HACE FALTA UN AÑO, DE QUE LAS INTERRUPCIONES SE PENALIZAN, ETC.") 

                print("According to our study...blablabla")
                print("For the algorithm we used, the minimum amount of samples is blablabla\n\n")


                print("Hope to see you soon! :)") 

            # User chooses: No 
            elif ans == "n":


                print("\n\nNice! Try to upload a greater amount or data and with less interruptions to be able to generate your personalized-AI glucose predictor!") 
                print("Hope to see you soon! :)")
            
            else: 
                print("Please enter a 'y' or 'n' to continue.")
