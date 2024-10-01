# Copyright (C) 2024 Antonio Rodriguez
# 
# This file is part of Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool.
# 
# Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool.  If not, see <http://www.gnu.org/licenses/>.

# DIY_top_module.py
# This script is the top moduleof  the Do It Yourself (DIY) app for personalized
# glucose prediction. After experimentation (see main_libreview.py),
# this function can be considered the DIY module itself ready to be used.
# This module is executed from a Docker image through the Dockerfile
# and data access and AI model generation is managed from there. Two possible scenarios
# are considered when calling this function from the terminal (and the proper Docker call, see README.md): 
#     
#     a) If it is the user's first use, it runs the normal training of the DL model following a
#     4-folds month-wise CV approach. The model that presents better performance is saved as a .h5 file.
#     Currently only LSTM model is implemented to save execution time and complexity. After this,
#     T1D-related personal data analysis an visualization is provided, and an 30' prediction is performed
#     and depicted using the last 24 hours of the user's data (96 data points at 15-minute intervals for
#     the LibreView data used to develop this framework). 

#     ***** THE CHOICE OF THE BEST MODEL MIGHT CHANGE IN SUBSEQUENT VERSIONS OF THIS FRAMEWORK *****
#     b) If the user has already a DL model, (i.e., step a) has been done once) the function loads the model,
#     performs the same analysis with the new updated user's data, and performs a new 1-hour prediction taking 
#     the last 24 hours of the user's data.
#
# For further detailes, please see the functions docummentation, the README.md file, and/or our paper. 
# "An AI-Based “Do-It-Yourself” Module for Interstitial Glucose Forecasting for People with Type 1 Diabetes"


import os 
# Avoid TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
import time
sys.path.append("..")
import warnings
import logging

# Custom libraries 
from app.your_data_read_and_analysis import *
from models.training import month_wise_multi_input_LibreView_4fold_cv, train_model, ISO_adapted_loss
from models.multi_step.LSTMVanilla import get_model as get_LSTM_multi_step
from evaluation.multi_step.evaluation import model_evaluation as multi_step_model_evaluation
from utils import get_LibreView_CGM_X_Y_multistep, generate_ranges_tags, generate_weights_vector
from app.app_visualization import *
from your_AI_DIY_parameters import *
from sensor_params import *

# Ignore warnings
warnings.filterwarnings("ignore")


# Avoid Python verbosity associated to TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# DIY module welcome message (web generated)
print("""\
 __          __    _                                _                                                 _____        _                            _ 
 \ \        / /   | |                              | |                                         /\    |_   _|      | |                          | |
  \ \  /\  / /___ | |  ___  ___   _ __ ___    ___  | |_  ___    _   _   ___   _   _  _ __     /  \     | | ______ | |__    __ _  ___   ___   __| |
   \ \/  \/ // _ \| | / __|/ _ \ | '_ ` _ \  / _ \ | __|/ _ \  | | | | / _ \ | | | || '__|   / /\ \    | ||______|| '_ \  / _` |/ __| / _ \ / _` |
    \  /\  /|  __/| || (__| (_) || | | | | ||  __/ | |_| (_) | | |_| || (_) || |_| || |     / ____ \  _| |_       | |_) || (_| |\__ \|  __/| (_| |
     \/  \/  \___||_| \___|\___/ |_| |_| |_| \___|  \__|\___/   \__, | \___/  \__,_||_|    /_/    \_\|_____|      |_.__/  \__,_||___/ \___| \__,_|
                                                                 __/ |                                                                            
                                                                |___/                                                                            
    """)
    
print("""\
  _____               _____  _         __     __                             _   __ 
 |  __ \             |_   _|| |        \ \   / /                            | | / _|
 | |  | |  ___  ______ | |  | |_  ______\ \_/ /___   _   _  _ __  ___   ___ | || |_ 
 | |  | | / _ \|______|| |  | __||______|\   // _ \ | | | || '__|/ __| / _ \| ||  _|
 | |__| || (_) |      _| |_ | |_          | || (_) || |_| || |   \__ \|  __/| || |  
 |_____/  \___/      |_____| \__|         |_| \___/  \__,_||_|   |___/ \___||_||_|  
                                                                                                                                                                  
    """)

print("""\    

   _____  _                                                       _  _        _               
  / ____|| |                                                     | |(_)      | |              
 | |  __ | | _   _   ___  ___   ___   ___   _ __   _ __  ___   __| | _   ___ | |_  ___   _ __ 
 | | |_ || || | | | / __|/ _ \ / __| / _ \ | '_ \ | '__|/ _ \ / _` || | / __|| __|/ _ \ | '__|
 | |__| || || |_| || (__| (_) |\__ \|  __/ | |_) || |  |  __/| (_| || || (__ | |_| (_) || |   
  \_____||_| \__,_| \___|\___/ |___/ \___| | .__/ |_|   \___| \__,_||_| \___| \__|\___/ |_|   
                                           | |                                                
                                           |_|                                                
    """)

print("Here, you will be able to have an in-depth analysis of your uploaded CGM data, as well as an AI-based prediction of your glucose level in the next 30 minutes!")
print("\nThis module has been validated, but it is still work in progress. Currently, the following sensor models are supported:")
print("\t- Freestyle Libre 2\n\t- FreeStyle LibreLink\n\t- LibreLink")

print("\nThe following input data is required (and supported) by this module:\n\t- CGM data (for now, at 15 minutes intervals)")

print("\nWe are currently working on:\n\t- Support for more sensor models\n\t- Adding more data, such as insuline, physical activity, etc.\n\t- More accurate predictions\n\t- More in-depth data analysis\n\t- More user-friendly interface\n\t- More user-friendly error messages\n\t- More user-friendly documentation\n\t- More user-friendly everything")

print("\n\n***IMPORTANT***: due to regulatory issues, this module (unfortunately) don't access to the sensor data in real time (yet), so every time that you want to have your prediction, you need to upload your most recent CGM data.")
print("***DISCLAIMER***: Please, do not use this tool as a replacement for professional medical advice, but always as a complementary tool to help you to manage your diabetes")

print("\nPlease, for any suggestions, feedback or bug reports, contact the developer at: antorguez95@hotmail.com\nYou can also visit the project's GitHub page at:")
print("https://github.com/antorguez95/Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool")

your_data_path = "/CGM_forecasting/drop_your_data_here_and_see_your_pred"

os.chdir(your_data_path)

if "your_AI_based_CGM_predictor.h5" not in os.listdir():

    # First of all, the users is asked about the directory where the data is stored
    print("\n\n\nSeems that it is your first time! It's nice having you here!")
    print("Let's begin!\n1) If you haven't download your CGM data, please do it!")
    print("2) Once you have your data, please, drop it in the /drop_your_data_here_and_see_your_pred. Otherwise, we will not be able to generate your personalized AI-model!\n")
    
    # Then, since the prenprocesisng and the AI architecture depends on the sensor, the user is asked about the sensor he or she is using 
    print("\nNow, please, type the letter corresponding to the glucose sensor model you are using:\n")
    print("a) Abbott - Freestyle Libre 2\nb) Abbott - FreeStyle LibreLink\nc) Abbott - LibreLink\nd) Other (not supported yet)")
    print("***ONLY Abbott sensors have been validated in this version of the tool.***") 

    ans = input()

    if ans == "a":
        sensor = libreview_sensors
        sensor_name = "Freestyle Libre 2"
    elif ans == "b":
        sensor = libreview_sensors
        sensor_name = "FreeStyle LibreLink"
    elif ans == "c":
        sensor = libreview_sensors
        sensor_name = "LibreLink"
    else:
        sys.exit("\nSorry, but this tool does not support your sensor. We are currently working on the inclusion of more sensors.")

    print("Nice! Could you confirm that your glucose sensor is a " + sensor_name+ "? Just to double check! (y/n)")
    ans = input()

    if ans == "y":
        pass
    else:
        sys.exit("\nSorry, we need to double check your sensor. Try again and make sure that your sensor is supported by this tool!")

    # Now, the user is asked about the unit of the glucose data
    print("\nNow, please, type 'a' if the unit of your glucose data is in 'mg/dL' or, 'b' if it is in 'mmol/L' instead")
    ans = input()

    if ans == "a":
        unit = "mg/dL"
    elif ans == "b":
        unit = "mmol/L"
    else:
        sys.exit("\nSorry, but you introduced a wrong letter. Please, try again type only 'a' or 'b'")

    print("\nSo, your glucose data is in " + unit + ", are we right? (y/n)")
    ans = input()
    if ans == "y":
        print("\nGreat! Let's move on!")
    else:
        sys.exit("\nOh, sorry! Try again and make sure you introduce the right unit! We'll be waiting for you!")
    
    # Set Keras to float 64 to work with the ISO-loss function
    tf.keras.backend.set_floatx('float64')

    # Save unit for further uses
    np.save('unit.npy', unit)

    print("You don't have your personalized-AI glucose predictor yet.\nNow, your data will be analyzed.")
    print("If your data does not contain many interruptions, and the CGM samples are enough, your personalized-AI glucose predictor will be generated using 1 year from your recently uploaded data.") 

    # Analyze the data
    print("\n\nAnalyzing your data... This could take a few seconds.\n\n")

    data_suitability, time_between_readings = get_your_oldest_year_npys_from_LibreView_csv(your_data_path, True)

    # Read the generated dictionary to save the sensor model 
    with open('libreview_data_1yr_recordings.pickle', 'rb') as handle:
        sensor_1yr_data = pickle.load(handle)

    for key in sensor_1yr_data.keys():
        for key2 in sensor_1yr_data[key].keys():
            for key3 in sensor_1yr_data[key][key2]:
                for key4 in sensor_1yr_data[key][key2][key3]:
                    for key5 in sensor_1yr_data[key][key2][key3][key4]:
                        your_sensor_model = (list(sensor_1yr_data[key][key2][key3][key4][key5].keys())[0])
    
    # Avoid errors if the sensor model  is not saved (meaning there is not 1 year of data)
    if len(list(sensor_1yr_data.keys())) == 0:
        pass
    else: 
        # Save in a npy file the sensor model 
        np.save('your_sensor_model.npy', your_sensor_model)

    # If data is suitable for AI:
    if data_suitability:
    
        ######## Begin CODE OF THE IMPLEMENTED AI FRAMEWORK ########
        # TO CHANGE THIS, CHANGE/ADD A DICTIONARY IN your_AI_DIY_parameters.py, or change the parameters of the current dictionary
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

        print("Congrats! The data you provided is enough to generate and train your personalized-AI glucose predictor!\n")

        print("Before proceeding to the AI-model generation, would you like to know more about how your data will be used to generate your personalized-AI glucose predictor? (y/n):")
        ans = input()

        # Load user's data and the associated timestamps 
        recordings = np.load('oldest_1yr_CGM.npy')
        timestamps = np.load('oldest_1yr_CGM_timestamp.npy', allow_pickle=True)

        if ans == "y":


            # Compute days between the first and the last timestamp
            days = (timestamps[-1] - timestamps[0]).days

            # Computing time in range, hyper and hypo
            if unit == 'mg/dL':
                severely_hypo = recordings[recordings < 54]
                hypo = recordings[(recordings >= 54) & (recordings < 70)]
                normal = recordings[(recordings >= 70) & (recordings <= 180)]
                hyper = recordings[(recordings > 180) & (recordings <= 250)]
                severely_hyper = recordings[recordings > 250]

            elif unit == 'mmol/L':
                severely_hypo = recordings[recordings < 3.0]
                hypo = recordings[(recordings >= 3.0) & (recordings < 3.9)]
                normal = recordings[(recordings >= 3.9) & (recordings <= 10.0)]
                hyper = recordings[(recordings > 10.0) & (recordings <= 13.9)]
                severely_hyper = recordings[recordings > 13.9]
            
            # Compute percentages
            # Compute time in range (TiR), Time above the range and time below the range 
            # Get the number of total samples 
            total_samples = len(recordings)

            # Get the TiR (number of samples in normal glycaemic level) # HARD-CODED WITH mg/dL
            TiR_samples = len(recordings[(recordings >= 70) & (recordings <= 180)])

            # Get the number of samples in the range above normal range 
            above_range_samples = len(recordings[recordings > 180])

            # Get the number of samples in the range below normal range
            below_range_samples = len(recordings[recordings < 70])

            # Compute the percentage of TiR
            TiR_percentage = (TiR_samples / total_samples) * 100
            above_range_percentage = (above_range_samples / total_samples) * 100
            below_range_percentage = (below_range_samples / total_samples) * 100

            # Generate text with only 2 decimal positions 
            tir_text = "{:.2f}%".format(TiR_percentage)
            above_range_text = "{:.2f}%".format(above_range_percentage)
            below_range_text = "{:.2f}%".format(below_range_percentage)

            print("\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")	
            print("\n\nYour glucose sensor model: " + sensor_name)
            print("Time between consecutive glucose readings: " + str(sensor["SAMPLE_PERIOD"])+ " minutes.")
            print("Your glucose data unit: " + unit + ".")
            print("\n\nYour uploaded data contains " + str(days) + " days of CGM data, from "  + str(timestamps[0]) +" to " +str(timestamps[-1]) + ".")
            print("To train your personalized AI model, we will take one year counting from " + str(timestamps[0]) + ".")
            print("\n\nThese are the times you have spent on each range:\n\t TAR: "+ tir_text +  " %\n\t TIR: " + above_range_text + " %\n\t TBR: "+ below_range_text + " %\n")

            print("If your sensor were perfect and would never stop reading (this never happens!), in one year you would have had " + str(int(365*24*(60/sensor["SAMPLE_PERIOD"]))) + " CGM readings.")
            print("However, we have found that, due to some reading interruptions, the number of available CGM readings was " +  str(len(recordings)) + ".")

            print("\n\nWith your CGM data, we could extract " +  str(len(recordings)) + " training samples, which is enough to generate your personalized-AI glucose predictor!")

            # print("\n\nIf you want to check a graphical summary of your data, check your_cgm_data_summary.png in the /drop_your_data_here_and_see_your_pred folder.")

            print("\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print("\n\n\nPlease, type 'next' once you have finished looking around your data to move to your AI glucose predictor")
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
        
        os.chdir("..")

        # If the directory "Your_AI_CGM_predictor" is not created, create it
        if "Your_AI_CGM_predictor" not in os.listdir():
                os.mkdir("Your_AI_CGM_predictor")
            
        # Get into the fold directory
        os.chdir("Your_AI_CGM_predictor")

        # Generating the X and Y to train the AI model. 
        X, Y, X_times, Y_times, num_blocks = get_LibreView_CGM_X_Y_multistep(recordings, timestamps, libreview_sensors, N, step, PH, plot = True, verbose = 0) 

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

        # Go back to "drop_your_data_and_see_your_pred" directory, where the model will be called further times and save model
        os.chdir("../../../drop_your_data_here_and_see_your_pred")
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

        print("\n\n\nSorry, but the data you provided is not enough to generate and train your personalized-AI glucose predictor.")
        print("Please, try again uploading more CGM readings containing at least 1 year of data.")
        print("If you think that this is an error, please contact the developer/maintainer at: antorguez95@hotmail.com")

        print("\n\nDo you want to know more about why did your data not meet the criteria to generate your personalized-AI glucose predictor? (y/n): ")
        ans = input()

        if ans == "y": 

            # Information about WHY WE CAN'T (we should include here the threshold of samples too). Currently it only consideres the year of data. 
            print("\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")	

            print("\n\nYour uploaded data contains only " + str(time_between_readings.days) + " days of CGM data")
            print("Currently, to train your personalized AI model, we should take one year counting from your first CGM reading.")

            print("Unfortunately, we have found that your data does not meet the criteria to generate your personalized-AI glucose predictor. You uploaded less than 1 year of CGM data.")

            # print("\n\nIf you want to check a graphical summary of your data, check your_cgm_data_summary.png in the /drop_your_data_here_and_see_your_pred folder.")

            print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print("Hope to see you soon! :)") 

        # User chooses: No 
        elif ans == "n":

            print("\n\nNice! Try to upload a greater amount or data and with less interruptions to be able to generate your personalized-AI glucose predictor!") 
            print("Hope to see you soon! :)")
        
        else: 
            print("Please enter a 'y' or 'n' to continue.")
    
# Now, if the .h5 is generated, we have to check if we have the full sequence of one day to generate the module. If not, inform the user
elif 'your_AI_based_CGM_predictor.h5' in os.listdir():

    print("\n\n\nWelcome again! It's always nice having you here!\n\n")
    print("You already have your personalized-AI glucose predictor!\nYou are able to have an 30' AI-based prediction of your glucose levels!")
    print("Remember that this prediction is based on your CGM data from your last 24 hours!")

    print("\n\n***IMPORTANT***: due to regulatory issues, this module (unfortunately) don't access to the sensor data in real time, so every time that you want to have a 1-hour prediction, you need to upload your CGM data.")
    print("***DISCLAIMER***: Please, do not use this tool as a replacement for professional medical advice, but always as a complementary tool to help you manage your T1D")

    print("\n\n\nExtracting the data from your last 24 hours...")

    # Load unit 
    unit = str(np.load('unit.npy'))

    
    # From the recently uploaded data, keys to access data are axtracted
    for file in os.listdir():
        if file.endswith(".csv"):
            filename = file
            break
    your_id = filename.split("_")[0][2:] 
    your_s = filename.split("_")[1][1:]
    your_r = filename.split("_")[2][1:]
    your_date = filename.split("_")[4][:-4]

    your_keys = [your_id, your_s, your_r, your_date] 

    # The (new uploaded) data is read and analyzed from the same folder than the first time 
    last_day_of_data, last_day_of_data_timestamps, full_sequence  = get_and_check_last_day_of_data(your_keys, your_data_path) 

    # Back to data path 
    os.chdir(your_data_path)

    # IF EVERYTHING IS OK, PROCEED TO THE PREDICTION
    if full_sequence:

        # Load your sensor model
        sensor_model = str(np.load('your_sensor_model.npy', allow_pickle=True))
        
        # Get the maximum and minimum values of the training to normalize it, so X is loaded.
        npy_name = 'X_'+your_id+'_1_'+sensor_model+'_CGM.npy'
        X = np.load(npy_name) # Harcoded: check 

        # Get the maximum and minimum of X
        max = np.max(X)
        min = np.min(X)

        # Data normalization
        data_norm = (last_day_of_data - min)/(max - min)

        # Get 1st derivative of the last day of data
        der = np.diff(data_norm)

        # Add the last point of derivative on the right of the array to have same dimension than the original array
        der = np.insert(der, -1, der[-1])

        # Stack both vectors
        model_input = np.dstack((data_norm, der))
        
        print("\n\nNice! You don't have CGM readings interruptions in the last 24 hours!")
        
        # Go to folder where the model is saved 
        # os.chdir("Your_AI_CGM_predictor")
        
        # Call the model
        print("\n\nLoading your personalized-AI glucose predictor...")
        model = tf.keras.models.load_model('your_AI_based_CGM_predictor.h5', custom_objects={'ISO_adapted_loss': ISO_adapted_loss})

        # Make the prediction
        print("\n\nPredicting your glucose levels...")
        prediction = model.predict(model_input)

        # Denormalize the prediction
        denorm_prediction = prediction*(max - min) + min

        # Number of predicted points computed directly from prediction 
        predicted_points = len(prediction[0])

        # Detect hypoglycemia and hyperglycemia
        # Check if any of the values is above or below range. Otherwise, it is normal
        # Default values
        hypoglycemia = False
        hyperglycemia = False

        if True in (np.unique(denorm_prediction) > 180):
            hyperglycemia = True
        elif True in (np.unique(denorm_prediction) < 70):
            hypoglycemia = True
        else : 
            hypoglycemia = False
            hyperglycemia = False
        
        # Load the RMSE
        rmse = np.load('rmse.npy')

        # Plot and save the prediction graphics 
        get_prediction_graphic(last_day_of_data, data_norm, predicted_points, last_day_of_data_timestamps, rmse, unit, prediction)

        print("\n\n\n")
        
        # Alert messages 
        if hyperglycemia:
            print("Watch out! According to your personalized-AI glucose predictor, you are at risk an HYPERGLYCAEMIA in the next 30'!")
        
        elif hypoglycemia:
            print("Watch out! According to your personalized-AI glucose predictor, you are at risk an HYPOGLYCAEMIA in the next 30'!")
        else:  
            print("Good! According to your personalized-AI glucose predictor, your glucose levels will remain in range in the next 30'!")
            print("Go to /drop_your_data_here_and_see_your_pred to see your prediction!")
            print("You can also type 'explorer your_path/drop_your_data_here_and_see_your_pred/your_last_prediction.png' to check it from here!")
            print("\nRemember that this is only an estimation! Don't take it as an absolute truth! :)")

    if not full_sequence:
        print("\nOops! We have detected some interruptions in your glucose sensor data in the last 24 hours!")
        print("\nOr maybe you just changed your sensor since you generated your personalized-AI glucose predictor.")
        print("Unfortunately, we cannot provide you with a prediction now :(")

        print("\nIf it is the latter, please remove all the content of the 'drop_your_data_here_and_see_your_pred' folder, drop a new file with a year of data with the new sensor and execute the docker command again!.")
        print("Please, try again later!")

