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
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from matplotlib import pyplot as plt
import pickle

from arch_params import *

def extract_Mikael_data(json_file_path : str, json_file_name : str, ONLY_CGM: bool = True): 
    """
    This function prepares the data from the json file (Mikael personal data) and returns
    a list of dictionaries containing data of interet for Type 1 Diabetes managament of
    different nature. 
    From the original json file, the following data is not extracted: 
        - HBA1C (empty)
        - Heart rate (empty)
        - Long acting data (empty)
        - Places (empty)
        - Transports (empty)
        - Stats : only basal rate is placed
        - Notes: skipped, need to be processed with something such as NLP
        - Photos: skipped, need to be processed with an image recognition approach
        - 'profiles' of Pump Events are not exctracted. Need to be discussed
        - 'raw' from Sleep data need to be studied and discussed

    Args:
    -----
        json_file_path: path to the json file
        json_file_name: name of the json file

    Returns:
    --------
        basa_data_dict: dictionary containing basal data through time
        blood_pressure_dict: dictionary containing blood pressure data through time
        bolus_data_dict: dictionary containing bolus data through time
        exercise_data_dict: dictionary containing exercise data through time
        carbohydrates_data_dict: dictionary containing carbohydrates data through time
        pump_events_data_dict: dictionary containing pump events data through time
        sgv_data_dict: dictionary containing glucose data through time
        sleep_data_dict: dictionary containing sleep data through time
        smbg_data_dict: dictionary containing glucose data through time
        steps_data_dict: dictionary containing steps data through time
        sleep_data_dict: dictionary containing sleep data through time
        weight_data_dict: dictionary containing weight data through time    
    
    """

    # Go to the dataset directory
    os.chdir(json_file_path)

    # Read the json file 
    with open(json_file_name) as f:
        data = json.load(f)
    
    # Set column names
    column_names = []

    for keys in data.keys():
        column_names.append(keys)

    # Create empty dataframe with the column names
    data_df = pd.DataFrame(columns=column_names)

    # Set 0 to the first row 
    data_df.loc[0] = 0

    # Declaration of empty dictionaries
    basal_data_dict = {}
    blood_pressure_dict = {}
    bolus_data_dict = {}
    exercise_data_dict = {}
    carbohydrates_data_dict = {}
    pump_events_data_dict = {}
    sgv_data_dict = {}
    sleep_data_dict = {}
    smgb_data_dict = {}
    steps_data_dict = {}
    weight_dict = {}

    # Read only "sgv" data if ONLY_CGM is set to True
    if ONLY_CGM == True:
        print("Reading sgv data...")

        # Add keys to bolus_data_dict
        for keys in data['sgv'].keys():
            sgv_data_dict[keys] = []

        for key in sgv_data_dict.keys():

            # Empty dataframe to be filled
            sgv_values_df = pd.DataFrame(columns=['time', 'glucose'])
            
            for i in range(len(data['sgv'][key])):     
                    sgv_values_df.loc[i] = [data['sgv'][key][i]['time'], data['sgv'][key][i]['glucose']]
            
            sgv_data_dict[key] = sgv_values_df
        
        # Save sgv data using pickle
        filename = 'CGM.pk1'
        with open(filename, 'wb') as f:
            pickle.dump(sgv_data_dict, f)
    
    # If not, read everything
    else:         

        ########################### BASAL DATA ########################### 
        print("Reading basal data...")
        # Declare a dictionary to store basal data

        # Add keys to basal_data_dict
        for keys in data['basal'].keys():
            basal_data_dict[keys] = []

        for key in basal_data_dict.keys():

            # Empty dataframe to be filled
            basal_values_df = pd.DataFrame(columns=['time', 'measurement'])

            for i in range(len(data['basal'][key])):     
                basal_values_df.loc[i] = [data['basal'][key][i]['time'], data['basal'][key][i]['amount']]
                
            basal_data_dict[key] = basal_values_df
        
        ########################### BLOOD PRESSURE ###########################
        print("Reading blood pressure data...")

        blood_pressure_dict = data['bloodPressure']

        ########################### BOLUS DATA ###########################
        print("Reading bolus data...")

        # Add keys to bolus_data_dict
        for keys in data['bolus'].keys():
            bolus_data_dict[keys] = []

        for key in bolus_data_dict.keys():

            # Empty dataframe to be filled
            bolus_values_df = pd.DataFrame(columns=['time', 'measurement', 'mealBolus'])

            for i in range(len(data['bolus'][key])):     

                # When "mealBolus" entry is absent, we assume that its value is False
                if data['bolus'][key][i].get('mealBolus') == True: 
                    bolus_values_df.loc[i] = [data['bolus'][key][i]['time'], data['bolus'][key][i]['amount'], data['bolus'][key][i]['mealBolus']]
                else: 
                    bolus_values_df.loc[i] = [data['bolus'][key][i]['time'], data['bolus'][key][i]['amount'], False]
                
            bolus_data_dict[key] = bolus_values_df

        ########################### EXERCISE DATA ###########################
        print("Reading exercise data...") 

        # Add keys to bolus_data_dict
        for keys in data['exercises'].keys():
            exercise_data_dict[keys] = []

        for key in exercise_data_dict.keys():

            # Empty dataframe to be filled
            exercise_values_df = pd.DataFrame(columns=['duration', 'start_time', 'end_type', 'exercise_description', 'exercise_profile' , 'tracker_app'])
            for i in range(len(data['exercises'][key])):     
            
                # When "type" is not specified, we substituted by "?"
                if data['exercises'][key][i].get('type') == True: 
                    exercise_values_df.loc[i] = [data['exercises'][key][i]['duration'], data['exercises'][key][i]['time'], data['exercises'][key][i]['endTime'], data['exercises'][key][i]['type'],
                                            data['exercises'][key][i]['raw']['activityType'], data['exercises'][key][i]['raw']['application']['packageName']]
                else: 
                    exercise_values_df.loc[i] = [data['exercises'][key][i]['duration'], data['exercises'][key][i]['time'], data['exercises'][key][i]['endTime'], "?",
                                            data['exercises'][key][i]['raw']['activityType'], data['exercises'][key][i]['raw']['application']['packageName']]
            
            exercise_data_dict[key] = exercise_values_df
        
        ########################### CARDBOHYDRATES DATA ###########################
        print("Reading carbohydrates data...")

        # Add keys to bolus_data_dict
        for keys in data['nutrition'].keys():
            carbohydrates_data_dict[keys] = []

        for key in carbohydrates_data_dict.keys():

            # Empty dataframe to be filled
            carbohydrates_values_df = pd.DataFrame(columns=['time', 'carbohydrate'])
            
            for i in range(len(data['nutrition'][key])):     
                    carbohydrates_values_df.loc[i] = [data['nutrition'][key][i]['time'], data['nutrition'][key][i]['carbohydrate']]
            
            carbohydrates_data_dict[key] = carbohydrates_values_df
        
        ########################### PUMP EVENTS DATA ###########################
        print("Reading pump events data...")

        # Add keys to bolus_data_dict
        for keys in data['pumpEvents'].keys():
            pump_events_data_dict[keys] = []

        for key in pump_events_data_dict.keys():

            # Empty dataframe to be filled
            pump_events_values_df = pd.DataFrame(columns=['time', 'type', 'duration'])
            for i in range(len(data['pumpEvents'][key])):     
            
                # There are entries where "profiles" is not present (###### WE HAVE TO ASK ABOUT THIS ######) 
                if data['pumpEvents'][key][i].get('profiles') != None: 
                    
                    # Sometimes "duration" is not present
                    if data['pumpEvents'][key][i].get('duration') != None:
                        pump_events_values_df.loc[i] = [data['pumpEvents'][key][i]['time'], data['pumpEvents'][key][i]['type'], data['pumpEvents'][key][i]['duration']]
                    else: 
                        pump_events_values_df.loc[i] = [data['pumpEvents'][key][i]['time'], data['pumpEvents'][key][i]['type'], "?"]            
                
                else:     
                    # Sometimes "duration" is not present
                    if data['pumpEvents'][key][i].get('duration')!= None:
                        pump_events_values_df.loc[i] = [data['pumpEvents'][key][i]['time'], data['pumpEvents'][key][i]['type'], data['pumpEvents'][key][i]['duration']]
                    else: 
                        pump_events_values_df.loc[i] = [data['pumpEvents'][key][i]['time'], data['pumpEvents'][key][i]['type'], "?"]    

            pump_events_data_dict[key] = pump_events_values_df

        ########################### SGV DATA ########################### 
        print("Reading sgv data...")

        # Add keys to bolus_data_dict
        for keys in data['sgv'].keys():
            sgv_data_dict[keys] = []

        for key in sgv_data_dict.keys():

            # Empty dataframe to be filled
            sgv_values_df = pd.DataFrame(columns=['time', 'glucose'])
            
            for i in range(len(data['sgv'][key])):     
                    sgv_values_df.loc[i] = [data['sgv'][key][i]['time'], data['sgv'][key][i]['glucose']]
            
            sgv_data_dict[key] = sgv_values_df
            
        ########################### SLEEP DATA ###########################
        print("Reading sleep data...")  

        # Add keys to bolus_data_dict
        for keys in data['sleep'].keys():
            sleep_data_dict[keys] = []

        for key in sleep_data_dict.keys():
            # Empty dataframe to be filled
            sleep_values_df = pd.DataFrame(columns=['start_time', 'end_time', 'duration', 'tracking_app'])
            
            if data['sleep'][key] == []:
                sleep_values_df.loc[i] = [None, None, None, None]
            else: 
                # Sometimes 'application' entry is not present 
                if data['sleep'][key][0].get('raw').get('application') == None:
                    sleep_values_df.loc[i] = [data['sleep'][key][0]['time'], data['sleep'][key][0]['endTime'], data['sleep'][key][0]['duration'], None]
                else: 
                    sleep_values_df.loc[i] = [data['sleep'][key][0]['time'], data['sleep'][key][0]['endTime'], data['sleep'][key][0]['duration'], data['sleep'][key][0]['raw']['application']['packageName']]
            
            sleep_data_dict[key] = sleep_values_df

        ########################### SMBG DATA ###########################
        print("Reading smbg data...")

        # Add keys to bolus_data_dict
        for keys in data['smbg'].keys():
            smgb_data_dict[keys] = []

        for key in smgb_data_dict.keys():

            # Empty dataframe to be filled
            smgb_values_df = pd.DataFrame(columns=['time', 'glucose'])
            
            for i in range(len(data['smbg'][key])):     
                    smgb_values_df.loc[i] = [data['smbg'][key][i]['time'], data['smbg'][key][i]['glucose']]
            
            smgb_data_dict[key] = smgb_values_df

        
        ########################### STEPS DATA ###########################
        print("Reading steps data...")

        # Add keys to bolus_data_dict
        for keys in data['steps'].keys():
            steps_data_dict[keys] = []

        for key in steps_data_dict.keys():

            # Empty dataframe to be filled
            steps_values_df = pd.DataFrame(columns=['start_time', 'end_time', 'steps'])
            
            for i in range(len(data['steps'][key])):     
                    steps_values_df.loc[i] = [data['steps'][key][i]['time'], data['steps'][key][i]['endTime'], data['steps'][key][i]['steps']]
            
            steps_data_dict[key] = steps_values_df

        ########################### WEIGHT DATA ###########################
        print("Reading weight data...")

        weight_dict = data['weight']
    
    return data , basal_data_dict, blood_pressure_dict, bolus_data_dict, exercise_data_dict, carbohydrates_data_dict, pump_events_data_dict, sgv_data_dict, sleep_data_dict, smgb_data_dict, steps_data_dict, weight_dict 

def get_CGM_X_Y_slow(CGM_data_dict: Dict, glucose_sensor : Dict, N: int, tau: int,
                prediction_time : int, 
                experiments_folder : str, verbose = int) -> Tuple[np.array, np.array, np.array, np.array]:
    
    """Generates the X and Y arrays for the Mikael T1DM dataset. The X array
    contains windows of size N and stride tau of the envelograms of the
    recordings sampled at 50 Hz, as done in Renna et al. [2]. The S array
    contains the heart state sequence of the recordings.

    Args:
    -----
        CGM_data_dict (dict): dictionary containing all the CGM entries of the json file
        N (int): window size
        tau (int): window stride
        prediction_time (int): prediction time in minutes
        experiments_folder (str): path to the folder where the experiments generated
        results will be saved
        verbose (int): Verbosity level.
    
    Returns:
    --------
        X (np.ndarray): 2D array with the windows of the CGM readings. Its shape
        is (number of windows, N).
        Y (np.ndarray): 1D array with the value just after the end of the correspondant window
        (Value Nth+1). Its shape is (number of windows, 1).
        X_times (np.array): datetime datatypes associated to X 
        Y_times (np.array): datetime datatypes associated to Y
    """


    # Set current directory as parent directory (code has been designed to have 'modular' paths)
    parent_directory = os.getcwd()

    # Create folder to save the experiments data
    if not os.path.exists(parent_directory+experiments_folder):
        os.makedirs(parent_directory+experiments_folder)

    os.chdir(parent_directory+experiments_folder)

    # Create a folder dependent on N, stride and step to save the data
    experiment_path = r'\N%i_stride%i_step%i' % (N, tau, prediction_time)
    if not os.path.exists(parent_directory+experiments_folder+experiment_path):
        os.makedirs(parent_directory+experiments_folder+experiment_path)

    # Change to that directory 
    os.chdir(parent_directory+experiments_folder+experiment_path)

    # Declaration of input, output and associated times (as datetime objects)
    X = np.zeros((0, N), dtype=np.float32)
    Y = np.zeros((0, round(glucose_sensor['PREDICTED_POINTS'])), dtype=np.float32) # Check values on sensor_params.py and arch_params.py
    X_times = np.zeros((0, N))
    Y_times = np.zeros((0, round(glucose_sensor['PREDICTED_POINTS']))) # Check values on sensor_params.py and arch_params.py

    # Variable to store CGM and times as a single continue value to further split them as desired
    time_concat = pd.DataFrame()
    glucose_concat = np.array(0)

    # Declare X an Y vector with all time and glucose concatenated data to further processing 
    for key in CGM_data_dict.keys():

        # Concatenate CGN and time with current key data
        time_concat = pd.concat([time_concat, CGM_data_dict[key]['time']], axis=0)
        glucose_concat = np.hstack((glucose_concat, CGM_data_dict[key]['glucose']))
    
    # Remove first element of the vector Y(0)
    glucose_concat = np.delete(glucose_concat, 0)

    # X to datetime format to further processing
    time_concat_date = pd.to_datetime(time_concat[0])

    # Reset index of to avoid further errors
    time_concat_date = time_concat_date.reset_index(drop=True)

    # Compute the differentce between two consecutive samples (in minutes)
    time_diff = time_concat_date.diff().dt.total_seconds().div(60, fill_value=0).astype(int)

    # Plot all time intervals between two consecutive samples
    # Set IEEE style
    plt.style.use(['science', 'ieee'])

    plt.plot(time_diff)

    # Set X label
    plt.xlabel('Sample difference')

    # Set Y label
    plt.ylabel('Minutes between sensor readings')

    # Save figure
    plt.savefig('sample_difference.png', dpi=300, bbox_inches='tight')

    # Find indexes where the difference between two consecutive samples is greater than 10 minutes
    time_diff_10_idx = np.where(time_diff >= 10)

    # Number of blocks without time intervals of more than 10 minutes
    n_blocks = len(time_diff_10_idx[0])
    print("Number of blocks is %i\n" % n_blocks)

    # Step for the output  value identification - 1: the output is 5 min (value 50) / 2: 10 min (value 51) / etc. 
    step = prediction_time

    # Global index useful to extract the blocks for the original array
    z = 0 # 1 in matlab

    # Numpy array to count samples in each block
    num_samples = np.zeros((n_blocks, 1))

    for i in range(0, n_blocks):

        print("Extracting data from Block %i/%i" % (i+1, n_blocks))

        # Extract time and glucose data from the block interval
        time_block = time_concat_date[z : time_diff_10_idx[0][i]]
        glucose_block = glucose_concat[z : time_diff_10_idx[0][i]]

        # Compute size of the current block
        block_size = (len(time_block))
        
        if verbose == 1:
            print("Block size is %i" % (block_size))
        
        # Loop until the last value possible value of the block considering N
        for j in range(0, round(block_size - N - step)):

            # Reference value for the initial data point to be collected 
            ini = j

            # Reference value for the last data point to be collected 
            end = ini + N

            # Extract glucose and time values of current block
            glucose_block = glucose_concat[ini : end]
            time_block = time_concat_date[ini : end]

            # Extract glucose output value (Ground Truth) and the time associated
            glucose_output = glucose_concat[round(end + step)]
            time_output = time_concat_date[round(end + step)]

            # Concatenate the current block to the X and Y vectors
            X = np.vstack((X, glucose_block))
            Y = np.vstack((Y, glucose_output)) 
            X_times = np.vstack((X_times, time_block)) 
            Y_times = np.vstack((Y_times, time_output))

            # Count the samples of the current block
            num_samples[i] = j+1
        
        # Print number of samples 
        if verbose == 1:
            print("Number of samples in block %i is %i\n" % ((i+1), num_samples[i]))

        # Update the global index
        z = time_diff_10_idx[0][i] 

        # Save training dataset summary in a txt file
        with open('dataset_summary.txt', 'w') as f:
            f.write('N = {}\n'.format(N))
            f.write('step = {}\n'.format(tau))
            f.write('prediction time = {}\n'.format(prediction_time))
            f.write('sensor type = {}\n'.format(glucose_sensor['NAME']))
            f.write('nº blocks = {}\n'.format(n_blocks))
        
        # Export X, Y and associated times as .npy files
        np.save('X.npy', X)
        np.save('Y.npy', Y)
        np.save('X_times.npy', X_times)
        np.save('Y_times.npy', Y_times)

    return X, Y, X_times, Y_times

def get_CGM_X_Y(CGM_data_dict: Dict, glucose_sensor : Dict, N: int, tau: int,
                prediction_time : int, 
                experiments_folder : str, 
                plot : bool, verbose = int) -> Tuple[np.array, np.array, np.array, np.array]:
    """Generates the X and Y arrays for the Mikael T1DM dataset. The X array
    contains windows of size N and stride tau of the envelograms of the
    recordings sampled at 50 Hz, as done in Renna et al. [2]. The S array
    contains the heart state sequence of the recordings.

    Args:
    -----
        CGM_data_dict (dict): dictionary containing all the CGM entries of the json file
        N (int): window size
        tau (int): window stride
        prediction_time (int): prediction time in minutes
        experiments_folder (str): path to the folder where the experiments generated
        results will be saved
        plot (bool): if True, plots the sample difference between two consecutive samples
        verbose (int): Verbosity level.
    
    Returns:
    --------
        X (np.ndarray): 2D array with the windows of the CGM readings. Its shape
        is (number of windows, N).
        Y (np.ndarray): 1D array with the value just after the end of the correspondant window
        (Value Nth+1). Its shape is (number of windows, 1).
        X_times (np.array): datetime datatypes associated to X 
        Y_times (np.array): datetime datatypes associated to Y
    """

    # Set current directory as parent directory (code has been designed to have 'modular' paths)
    parent_directory = os.getcwd()

    # Create folder to save the experiments data
    if not os.path.exists(parent_directory+experiments_folder):
        os.makedirs(parent_directory+experiments_folder)

    os.chdir(parent_directory+experiments_folder)

    # Create a folder dependent on N, stride and step to save the data
    experiment_path = r'\N%i_stride%i_step%i' % (N, tau, prediction_time/5)
    if not os.path.exists(parent_directory+experiments_folder+experiment_path):
        os.makedirs(parent_directory+experiments_folder+experiment_path)

    # Change to that directory 
    os.chdir(parent_directory+experiments_folder+experiment_path)

    # Variable to store CGM and times as a single continue value to further split them as desired
    time_concat = np.array(0)
    glucose_concat = np.array(0)

    # Declare X an Y vector with all time and glucose concatenated data to further processing 
    for key in CGM_data_dict.keys():

        # Concatenate CGN and time with current key data
        time_concat = np.hstack((time_concat, CGM_data_dict[key]['time']))
        glucose_concat = np.hstack((glucose_concat, CGM_data_dict[key]['glucose']))

    # Remove first element arrays
    glucose_concat = np.delete(glucose_concat, 0)
    time_concat = np.delete(time_concat, 0)

    # For dates to seconds in int64 for faster processing and accurate computation
    time_concat = time_concat.astype('datetime64[s]')
    time_concat = time_concat.astype('int64')

    # Compute the differentce between two consecutive samples (in minutes)
    time_diff = np.diff(time_concat)
    
    # Plot all time intervals between two consecutive samples
    if plot == True: 
    # Set IEEE style
        plt.style.use(['science', 'ieee'])

        plt.plot(time_diff/60)

        # Set X label
        plt.xlabel('Sample difference')

        # Set Y label
        plt.ylabel('Minutes between sensor readings')

        # Save figure
        plt.savefig('sample_difference.png', dpi=300, bbox_inches='tight')

    # Find indexes where the difference between two consecutive samples is greater than 10 minutes
    time_diff_10_idx = np.where(time_diff >= 600)

    # Number of blocks without time intervals of more than 10 minutes
    n_blocks = len(time_diff_10_idx[0])
    print("Number of blocks is %i\n" % n_blocks)

    # Step for the output  value identification - 1: the output is 5 min (value 50) / 2: 10 min (value 51) / etc. 
    step = prediction_time/5

    # Global index useful to extract the blocks for the original array
    global_idx = 0 # 1 in matlab

    # Numpy array to count samples in each block
    num_samples = np.zeros((n_blocks, 1))

    # List to store the indexes X and Y (faster computation than concatenate arrays)
    X_init_list = []
    X_end_list = []
    Y_idxs_list = []

    for i in range(0, n_blocks):
    
        # Compute size of the current block
        block_size = time_diff_10_idx[0][i]-global_idx
        
        if verbose == 1:
            print("Block size is %i" % (block_size))
        
        # Loop until the last value possible value of the block considering N
        for j in range(0, round(block_size - N - step )):

            # Reference value for the initial data point to be collected 
            X_init_list.append(global_idx+j)

            # Reference value for the last data point to be collected 
            X_end_list.append(global_idx+j+N)

            # Extract glucose output (Y) indexes (Ground Truth) and the time associated 
            Y_idxs_list.append(round(global_idx+j+N+step-1)) # For slicing N samples, index N. For indexing sample in the Nth, index N

            # Count the samples of the current block
            num_samples[i] = j+1

            # Print number of samples 
            if verbose == 1:
                print("Number of samples in block %i is %i\n" % ((i+1), num_samples[i]))

        # Update the global index
        global_idx = time_diff_10_idx[0][i] 

    # Declare X an Y vector with all time and glucose concatenated data to further processing
    X = np.zeros((len(X_init_list), N), dtype=np.float32)
    Y = np.zeros((len(Y_idxs_list), round(glucose_sensor['PREDICTED_POINTS'])), dtype=np.float32) # Check values on sensor_params.py and arch_params.py
    X_times = np.empty((len(X_init_list), N), dtype='datetime64[s]')
    Y_times = np.empty((len(Y_idxs_list), round(glucose_sensor['PREDICTED_POINTS'])), dtype='datetime64[s]')

    for i in range(0, X.shape[0]):
        X[i,:] = glucose_concat[X_init_list[i] : X_end_list[i]]
        Y[i,:] = glucose_concat[Y_idxs_list[i]]
        X_times[i,:] = time_concat[X_init_list[i] : X_end_list[i]]
        Y_times[i] = time_concat[Y_idxs_list[i]] 

    # Save training dataset summary in a txt file
    with open('dataset_summary.txt', 'w') as f:
        f.write('N = {}\n'.format(N))
        f.write('tau = {}\n'.format(tau))
        f.write('prediction time = {}\n'.format(prediction_time))
        f.write('sensor type = {}\n'.format(glucose_sensor['NAME']))
        f.write('nº blocks = {}\n'.format(n_blocks))

    # Export X, Y and associated times as .npy files
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    np.save('X_times.npy', X_times)
    np.save('Y_times.npy', Y_times) 

    return X, Y, X_times, Y_times

def get_CGM_X_Y_multistep(CGM_data_dict: Dict, glucose_sensor : Dict, N: int, tau: int,
                prediction_horizon : int, 
                experiments_folder : str, 
                plot : bool, verbose = int) -> Tuple[np.array, np.array, np.array, np.array]:
    """Generates the X and Y arrays for the Mikael T1DM dataset. The X array
    contains windows of size N and stride tau of the envelograms of the
    recordings sampled at 50 Hz, as done in Renna et al. [2]. The S array
    contains the heart state sequence of the recordings.

    Args:
    -----
        CGM_data_dict (dict): dictionary containing all the CGM entries of the json file
        N (int): window size
        tau (int): window stride
        prediction_time (int): prediction horizon in minutes
        experiments_folder (str): path to the folder where the experiments generated
        results will be saved
        plot (bool): if True, plots the sample difference between two consecutive samples
        verbose (int): Verbosity level.
    
    Returns:
    --------
        X (np.ndarray): 2D array with the windows of the CGM readings. Its shape
        is (number of windows, N).
        Y (np.ndarray): 1D array with the value just after the end of the correspondant window
        (Value Nth+1). Its shape is (number of windows, 1).
        X_times (np.array): datetime datatypes associated to X 
        Y_times (np.array): datetime datatypes associated to Y
    """

    # Set current directory as parent directory (code has been designed to have 'modular' paths)
    parent_directory = os.getcwd()

    # Create folder to save the experiments data
    if not os.path.exists(parent_directory+experiments_folder):
        os.makedirs(parent_directory+experiments_folder)

    os.chdir(parent_directory+experiments_folder)

    # Create a folder dependent on N, stride and step to save the data
    experiment_path = r'\N%i_stride%i_step%i' % (N, tau, prediction_horizon/5)
    if not os.path.exists(parent_directory+experiments_folder+experiment_path):
        os.makedirs(parent_directory+experiments_folder+experiment_path)

    # Change to that directory 
    os.chdir(parent_directory+experiments_folder+experiment_path)

    # Variable to store CGM and times as a single continue value to further split them as desired
    time_concat = np.array(0)
    glucose_concat = np.array(0)

    # Declare X an Y vector with all time and glucose concatenated data to further processing 
    for key in CGM_data_dict.keys():

        # Concatenate CGN and time with current key data
        time_concat = np.hstack((time_concat, CGM_data_dict[key]['time']))
        glucose_concat = np.hstack((glucose_concat, CGM_data_dict[key]['glucose']))

    # Remove first element arrays
    glucose_concat = np.delete(glucose_concat, 0)
    time_concat = np.delete(time_concat, 0)

    # For dates to seconds in int64 for faster processing and accurate computation
    time_concat = time_concat.astype('datetime64[s]')
    time_concat = time_concat.astype('int64')

    # Compute the differentce between two consecutive samples (in minutes)
    time_diff = np.diff(time_concat)
    
    # Plot all time intervals between two consecutive samples
    if plot == True: 
    # Set IEEE style
        plt.style.use(['science', 'ieee'])

        plt.plot(time_diff/60)

        # Set X label
        plt.xlabel('Sample difference')

        # Set Y label
        plt.ylabel('Minutes between sensor readings')

        # Save figure
        plt.savefig('sample_difference.png', dpi=300, bbox_inches='tight')

    # Find indexes where the difference between two consecutive samples is greater than 10 minutes
    time_diff_10_idx = np.where(time_diff >= 600)

    # Number of blocks without time intervals of more than 10 minutes
    n_blocks = len(time_diff_10_idx[0])
    print("Number of blocks is %i\n" % n_blocks)

    # Step for the output  value identification - 1: For N = 49, the output is 5 min (value 50) / 2: 10 min (value 51) / etc. 
    step = round(prediction_horizon/5)

    # Global index useful to extract the blocks for the original array
    global_idx = 0 # 1 in matlab

    # Numpy array to count samples in each block
    num_samples = np.zeros((n_blocks, 1))

    # List to store the indexes X and Y (faster computation than concatenate arrays)
    X_init_list = []
    X_end_list = []
    Y_init_list = []
    Y_end_list = []

    for i in range(0, n_blocks):
    
        # Compute size of the current block
        block_size = time_diff_10_idx[0][i]-global_idx
        
        if verbose == 1:
            print("Block size is %i" % (block_size))
        
        # Loop until the last value possible value of the block considering N
        for j in range(0, round(block_size - N - step)):

            # Reference value for the initial data point to be collected 
            X_init_list.append(global_idx+j)

            # Reference value for the last data point to be collected 
            X_end_list.append(global_idx+j+N)

            # Reference value for the initial Y point to be collected 
            Y_init_list.append(global_idx+j+N)

            # Reference value for the last data point to be collected 
            Y_end_list.append(global_idx+j+N+step)

            # Extract glucose output (Y) indexes (Ground Truth) and the time associated 
            # Y_idxs_list.append(round(global_idx+j+N+step-1)) # For slicing N samples, index N. For indexing sample in the Nth, index N

            # Count the samples of the current block
            num_samples[i] = j+1

            # Print number of samples 
            if verbose == 1:
                print("Number of samples in block %i is %i\n" % ((i+1), num_samples[i]))

        # Update the global index
        global_idx = time_diff_10_idx[0][i] 

    # Declare X an Y vector with all time and glucose concatenated data to further processing
    X = np.zeros((len(X_init_list), N), dtype=np.float32)
    Y = np.zeros((len(Y_init_list), round(prediction_horizon/5)), dtype=np.float32) # Check values on sensor_params.py and arch_params.py
    X_times = np.empty((len(X_init_list), N), dtype='datetime64[s]')
    Y_times = np.empty((len(Y_init_list), round(prediction_horizon/5)), dtype='datetime64[s]')

    for i in range(0, X.shape[0]):
        X[i,:] = glucose_concat[X_init_list[i] : X_end_list[i]]
        Y[i,:] = glucose_concat[Y_init_list[i] : Y_end_list[i]]
        X_times[i,:] = time_concat[X_init_list[i] : X_end_list[i]]
        Y_times[i] = time_concat[Y_init_list[i] : Y_end_list[i]] 

    # Save training dataset summary in a txt file
    with open('dataset_summary.txt', 'w') as f:
        f.write('N = {}\n'.format(N))
        f.write('tau = {}\n'.format(tau))
        f.write('prediction time = {}\n'.format(prediction_horizon))
        f.write('sensor type = {}\n'.format(glucose_sensor['NAME']))
        f.write('nº blocks = {}\n'.format(n_blocks))

    # Export X, Y and associated times as .npy files
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    np.save('X_times.npy', X_times)
    np.save('Y_times.npy', Y_times) 

    return X, Y, X_times, Y_times



