from typing import Dict, List 
import os
import pickle
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from models.training import ISO_adapted_loss
from libreview_utils import create_LibreView_results_dictionary
import json
from sensor_params import *
from evaluation.multi_step.evaluation import model_evaluation as multi_step_model_evaluation
from utils import get_LibreView_CGM_X_Y_multistep

set_of_libreview_keys = [["001", "001", "001", "2024"],
            ["003", "001", "001", "2024"],
            ["004", "001", "001", "2024"],
            ["007", "001", "001", "2024"], 
            ["008", "001", "001", "2024"],
            ["011", "001", "001", "2024"],
            #["013", "001", "001", "2024"],
            ["014", "001", "001", "2024"],
            ["015", "001", "001", "2024"],
            ["025", "001", "001", "2024"],
            ["026", "001", "001", "2024"],
            ["029", "001", "001", "2024"],
            ["039", "001", "001", "2024"],
            ["043", "001", "001", "2024"],
            ["045", "001", "001", "2024"],
            ["046", "001", "001", "2024"],
            ["048", "001", "001", "2024"],
            ["049", "001", "001", "2024"],
            ["051", "001", "001", "2024"],
            ["055", "001", "001", "2024"],
            ["057", "001", "001", "2024"],
            ["058", "001", "001", "2024"],
            ["060", "001", "001", "2024"],
            ["061", "001", "001", "2024"],
            ["062", "001", "001", "2024"],
            ["063", "001", "001", "2024"],
            ["065", "001", "001", "2024"],
            ["067", "001", "001", "2024"],
            ["068", "001", "001", "2024"]]


def read_test_csv(dataset_path : str, save_dict : bool = True) -> Dict : 

    """
    This function reads the .csv files from the dataset (downloaded by the endocrinologist)
    and stores them in a dictionary. This function takes a while. It returns a dictionary with 
    the CGM-related most important information. 

    Args:
    ----
        dataset_path: Path where the .csv files are stored. 
        save_dict: Flag to save the dictionary with pickle.
    
    Returns:
    -------
        data_dict: Dictionary with the CGM readings and timestamps for each subject, sensor and recording.
    
    """

    # Go to the dataset directory 
    os.chdir(dataset_path)

    # Create empty dictionary 
    data_dict = {}

    # Read all the available .csv files and store them in a dictionary 
    for file in os.listdir(dataset_path) : 

        # Only iterate on the .csv files that contains patient's data 
        if "ID" not in file :
            pass
        else: 

            print("Reading ", file, "...")
            
            # Extract the useful information from the file name to use them as dictionary keys 
            id = file.split("_")[0][2:]
            s = file.split("_")[1][1:]
            r = file.split("_")[2][1:]
            download_date = file.split("_")[4][:-4]

            # Create the dictionary for every patient, sensor and recording
            data_dict[id] = {}
            data_dict[id][s] = {}
            data_dict[id][s][r] = {}
            data_dict[id][s][r][download_date] = {}

            # Only read_csv is called if the file is .csv
            if file.endswith(".csv") : 
                
                # Read the .csv and store it in a DataFrame. 
                current_recordings = pd.read_csv(file, low_memory=False)#, encoding='latin-1')

                # Clean NaN values
                current_recordings = current_recordings.dropna(axis=0, subset=['Tipo de registro'])

                # Recording #14-01-01 has an error in the timestamps from sample 71870 to sample 74580. These are removed
                if id == "014" and s == "001" and r == "001" : 
                    idxs = np.where(current_recordings['Sello de tiempo del dispositivo'] == '01-01-0001 00:00')
                    current_recordings.drop(current_recordings.index[71870:74581], inplace=True)

                # Conver timestamps to datetime64
                current_recordings['Sello de tiempo del dispositivo'] = pd.to_datetime(current_recordings['Sello de tiempo del dispositivo'],
                                                                                    dayfirst=True,
                                                                                    format="%Y-%m-%d %H:%M",
                                                                                    exact=True)

                # Obtain sensors MACs (this is more robust that obtaining the sensor names, which has a lot of typos)
                MACs = current_recordings['Número de serial'].unique()

                # Iterate over the MACs, since it contains less errors than 'Dispositivo' column
                for i in range(0, len(MACs)) :

                    # Some instances (e.g., 014), brings NaN serial number (MAC). These are discarded and not considered in further steps
                    if MACs[i] is not np.nan : 

                        # Find the indices of the MACs
                        MAC_idxs = np.where(current_recordings['Número de serial'] == MACs[i])

                        # We take the first idx to obtain the sensor name 
                        sensor_name = current_recordings['Dispositivo'].iloc[MAC_idxs[0][0]]

                        # Empty arrays and DataFrames
                        empty_array = np.empty((1)) # to be filled with the readings separately
                        
                        # Create the dictionary for every recording, date and sensor
                        data_dict[id][s][r][download_date][MACs[i]] = {sensor_name : {"CGM" : {"reading" : np.empty((0), dtype=np.float64),
                                                                        "timestamp" : np.empty((0))},#, dtype='datetime64[s]')},
                                                            "Escanned CGM" : {"reading" : np.empty((0)),
                                                                        "timestamp" : np.empty((0))},#, dtype='datetime64[s]')},
                                                            "Insulin no num" : {"reading" : np.empty((0)),
                                                                        "timestamp" : np.empty((0))},# dtype='datetime64[s]')},
                                                            "Fast insulin" : {"reading" : np.empty((0)),
                                                                        "timestamp" : np.empty((0))},# dtype='datetime64[s]')}, 
                                                            "Food no num" : {"reading" : np.empty((0)),
                                                                        "timestamp" : np.empty((0))}}}#, dtype='datetime64[s]')}}}
            
                # Iterate over all the rerconding and place them and their timestamp in the corresopndant dictionary entry 
                for i in range(0,current_recordings.shape[0]): 

                    # Update current sensor name and MAC
                    curr_sensor_name = current_recordings['Dispositivo'].iloc[i]
                    curr_MAC = current_recordings['Número de serial'].iloc[i]
                    
                    # Depeding on the register type, some columns are useful and some are not 
                    register_type = round(current_recordings['Tipo de registro'].iloc[i])

                    match register_type:
                        case 0:  # Historial de glucosa mg/dL
                            # Add element to the dictionary
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["CGM"]["reading"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["CGM"]["reading"], current_recordings['Historial de glucosa mg/dL'].iloc[i])
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["CGM"]["timestamp"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["CGM"]["timestamp"], current_recordings['Sello de tiempo del dispositivo'].iloc[i])
                        case 1:  # Escaneo de glucosa mg/dL 
                            # Add element to the dictionary
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Escanned CGM"]["reading"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Escanned CGM"]["reading"], current_recordings['Escaneo de glucosa mg/dL'].iloc[i])
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Escanned CGM"]["timestamp"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Escanned CGM"]["timestamp"], current_recordings['Sello de tiempo del dispositivo'].iloc[i])
                        case 2:  # ¿¿¿¿¿¿ Insulina de acción rápida no numérica ?????
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Insulin no num"]["reading"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Insulin no num"]["reading"], current_recordings['Insulina de acción rápida no numérica'].iloc[i])
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Insulin no num"]["timestamp"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Insulin no num"]["timestamp"], current_recordings['Sello de tiempo del dispositivo'].iloc[i])
                        case 3:  # ¿¿¿¿¿¿ Insulina de acción rápida (unidades) ?????
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Fast insulin"]["reading"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Fast insulin"]["reading"], current_recordings['Insulina de acción rápida (unidades)'].iloc[i])
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Fast insulin"]["timestamp"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Fast insulin"]["timestamp"], current_recordings['Sello de tiempo del dispositivo'].iloc[i])
                        case 4:  # ¿¿¿¿¿¿ Alimento no numérico ?????
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Food no num"]["reading"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Food no num"]["reading"], current_recordings['Alimento no numérico'].iloc[i])
                            data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Food no num"]["timestamp"] = np.append(data_dict[id][s][r][download_date][curr_MAC][curr_sensor_name]["Food no num"]["timestamp"], current_recordings['Sello de tiempo del dispositivo'].iloc[i])                    
                        case 5:  # ¿¿¿¿¿¿ Carbohidratos (gramos) ?????
                            pass
                        case 6:  # ¿¿¿¿¿¿ Carbohidratos (porciones) ?????
                            pass

    # Save dictionary using pickle
    if save_dict : 
        filename = 'libreview_test_data.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dict

def get_end_training_dates(parent_dir : str, target_dir : str) -> Dict: 
    
    """ This function is for the final test of this work. After performing the 4-folds
    trimester wise cross-validation, more data were collected. This data were used for a 
    final validation of the models. Notice that this data is more recent that those used 
    in the previous step. Hence, to avoid use data that were used in the training process, 
    the last timestamps of the data were extracted. It returns a dictionary containing 
    the last timestamp of each subject. See other functions to understand the structure 
    of this dictionary. This function also saves such dictionary. 
    
    Args:
    ----
        dir : Path were the new data were stored. 
        target_dir : Path where the dictionary will be saved.
    
    Returns:
    -------
        subjects_end_trianing_dates: Dictionary with the last timestamps of the data for each subject (n=29 at this stage)
 
    """

    # Empty dict to be filled with subject - end dates pairs
    subjects_end_training_dates = {}

    # Parent directory where all timestamps are placed
    parent_dir = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files"

    os.chdir(parent_dir)

    # Iterate over the ID folders to fecht the end data for each subject 
    for id in os.listdir(): 

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('ISO' not in id) and ('test' not in id) and ('svg' not in id) and ('pickle' not in id):
                    
            # Go to the ID directory 
            os.chdir(id)
            
            # New dict entry with the ID
            subjects_end_training_dates[id] = {}

            # Construct path Hard-coded. (This directory should be the same if you are using the same functions)
            dir = r'C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files\{}\N96\step1\PH60\multi\month-wise-4-folds\norm_min-max\None_sampling\DIL-1D-UNET\ISO_loss'.format(id)

            # Go to the directory
            os.chdir(dir)

            # Read the .npy file with the timestamps 
            timestamps = np.load('Y_times.npy', allow_pickle=True)

            # Extact the most recent timestamp to compare to the new data to make sure that test is done with newer data 
            subjects_end_training_dates[id] = str(timestamps[-1][-1])

            # Back to parent directory 
            os.chdir(parent_dir)
    
    # Go to the test set directory 
    os.chdir(target_dir)

    # Save dictionary using pickle
    filename = 'subjects_end_training_dates.pickle' 
    with open(filename, 'wb') as handle:
        pickle.dump(subjects_end_training_dates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return subjects_end_training_dates


def extract_test_data_recordings (subjects_end_training_dates: Dict, data_dict : Dict) -> Dict: 

    """
    This function returns and save with Pickle the dictionary with the test data recordings after
    being processed by the prepare_LibreView_data function. It sorts the data by ID and by sensor in case
    a subject has more than one. The end dates of the training data should be provided. If the most recent 
    test data is older than the newest train data, this subject is discarded, since it does not provide a
    proper test set. 

    Args:
    ----
       subjects_end_training_dates: Dictionary with the last timestamps of the data for each subject (n=29 at this stage)
       data_dict: Dictionary with the CGM data after read the .csv files. 
    
    Returns:
    -------
        test_data_recordings: Dictionary with the test data recordings sorted by ID and sensor.  
    """

    # Fill the dictionary with the test data 
    test_data_recordings = {}

    # Iterate over all dictionary keys
    for i in range(0,len(set_of_libreview_keys)):

        # Initialize the dictionary entries
        test_data_recordings[set_of_libreview_keys[i][0]] = {}
        test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]] = {}
        test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]] = {}
        test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]] = {}

        for key in data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():

            # Initialize the entry regarding the MAC
            test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key] = {}

            for key2 in data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key].keys():

                # There could be without data 
                if not data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"].any():
                    pass
                else: 

                    # Calculate number of days so we can discard recordings with less than a year of data
                    data_1st_sample = data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"][0]
                    data_last_sample = data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"][-1]

                    # Discard data that are on the .csv but are previous to the end of the training period
                    if data_last_sample > np.datetime64(subjects_end_training_dates[set_of_libreview_keys[i][0]]):

                        # Corresponding CGM readings and their timestamps
                        cgm = data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"]
                        cgm_timestamp = data_dict[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"]

                        
                        # Fill dictionary with readings of at least one year
                        test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2] = {"CGM" : {"reading" : cgm,
                                                                                                                                            "timestamp" : cgm_timestamp}}
                    else: 
                        pass
                        # print(data_last_sample, np.datetime64(subjects_end_training_dates[set_of_libreview_keys[i][0]]))

    # Iterate over all dictionary keys to delete the entries that are empty (meaning that they had <1 year of data )
    for i in range(0,len(set_of_libreview_keys)):      
        for key in list(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys()):
        
                # Check if the entry is empty or not to delete it 
                if len(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]) == 0:

                    # Delete entry 
                    del test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]

    # Check if there are IDs completely empty to delete them
    for i in range(0,len(set_of_libreview_keys)):      
        if len(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]]) == 0:

            # Delete entry 
            del test_data_recordings[set_of_libreview_keys[i][0]]

    # Save dictionary as pickle
    with open('libreview_test_data_recordings.pickle', 'wb') as handle:
        pickle.dump(test_data_recordings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return test_data_recordings

def discard_data_from_sensor_black_list(test_data_recordings : Dict, sensor_black_list : List) -> Dict:
    
    """
    Since there are plenty of glucose sensors in the market and not all of them
    have the same sampling period, dataformat, etc., this function is introduced to 
    discard a given sensor model. Now, the criterion is to have a 15-min sampling period sensor.
    But this might change in the future. The input parameter is the sensor black list. There
    is not output, just a filtering of the previous dictionary. 

    Args:
    ----
        test_data_recordings: Dictionary with the test data recordings sorted by ID and sensor.
        sensor_black_list: List of sensors to be discarded from the dictionary.

    """

    # Iterate over all dictionary keys
    for i in range(0,len(set_of_libreview_keys)):
        for key in test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():
            
            # Extract sensor model to discard FreeStyle Libre 3, since it has a sampling period of 5 minutes
            sensor_model = list(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key].keys())[0]
            
            if sensor_model in sensor_black_list:
                # delet entry 
                test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key] = {}

    # Iterate over all dictionary keys to delete the entries that are empty (meaning that they had Free Style Libre 3 sensor)
    for i in range(0,len(set_of_libreview_keys)):      
        for key in list(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys()):
        
                # Check if the entry is empty or not to delete it 
                if len(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]) == 0:

                    # Delete entry 
                    del test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]
    
    return test_data_recordings


def get_ID_sensor_MAC_pairs(yr_data : Dict) -> Dict:
    """
    In this experiment, the data is done with just new test data. But it must be with 
    the same sensors used in the training and validation data for all subjects. Thus, 
    from a dictionary creates and returns in the training step, we extract a dictionry with the 
    IDs matched to the MACs of the sensors they used within the selected periods. This dictionary 
    will be used to filter the already generated test data.  

    Args:
    ----
        yr_data: Dictionary with the training and validation data recordings sorted by ID and sensor.
    
    Returns:
    -------
        ID_sensor_MAC_pairs: Dictionary with the IDs matched to the MACs of the sensors they used within the selected periods. 
    """

    # Declare a dictionary to store the MACs of the sensors used to trained the personalized models to filter the test data 
    data_with_1yr_sensor_MAC = {}
    for key in yr_data:
        for key2 in yr_data[key]: 
            for key3 in yr_data[key][key2]:
                for key4 in yr_data[key][key2][key3]:
                    for key5 in yr_data[key][key2][key3][key4]:
                        data_with_1yr_sensor_MAC[key] = key5
    
    return data_with_1yr_sensor_MAC


def filter_subjects_that_change_sensor(test_data_recordings : Dict, data_with_1_yr_sensor_MAC : Dict) -> Dict: 
    """
    This function filters the Dictionary that contain the test data recordings using the dicionaty that contains
    the MACs of the sensors used in the trainin and validation step. It returns a dictionary with the subjects 
    that did not change the sensor, or, at least, have data from the same sensor. The rest of the data entris 
    are deleted

    Args:
    ----
        test_data_recordings: Dictionary with the test data recordings sorted by ID and sensor.
        data_with_1_yr_sensor_MAC: Dictionary with the IDs matched to the MACs of the sensors they used within the selected periods.
    
    Returns: 
    -------
        test_data_recordings: Filtered dictionary with subjects that have data from the same sensor used in the training and validation step.

    """

    # Iterate over all dictionary keys
    for i in range(0,len(set_of_libreview_keys)):
        for key in test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():

            # Check if the MACs match. Only if they do, the test data will be used 
            if key == data_with_1_yr_sensor_MAC[set_of_libreview_keys[i][0]]:
                pass
            else:
                # Empty entry to further delete it 
                test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key] = {}
            
    # Iterate over all dictionary keys to delete the entries that are empty (meaning that they a subject changed his/her sensor in the test data set)
    for i in range(0,len(set_of_libreview_keys)):      
        for key in list(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys()):
        
                # Check if the entry is empty or not to delete it 
                if len(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]) == 0:

                    # Delete entry 
                    del test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]

    return test_data_recordings



def remove_training_data_from_test_set(test_data_recordings : Dict, subjects_end_training_dates : Dict,  verbose: int) -> Dict:
    
    
    """
    To test date solely with data from the same sensor (as previosly explained), and
    that were not used to train the models, the timestamps from the training steps are
    used to remove the test data that are older or equal than that timestamps (i.e., same data). 
    If desired, informative messages are printed. This function returns the CGM readings for all subjects
    with only the test data. 

    Args:
    ----
        test_data_recording: Dictionary with the test data recordings sorted by ID and sensor.
        subject_end_training_dates: Dictionary with the last timestamps of the data for each subject 
        verbose: Flag to print informative messages.
    
    Returns: 
    -------
        test_data_recording: Dictionary with the test data recordings sorted by ID and sensor with only the test data. 

    """

    # Convert subject's end training dates to datetime64 to compare
    for key in subjects_end_training_dates:
        subjects_end_training_dates[key] = np.datetime64(subjects_end_training_dates[key])
   
    # Concatenate data to be able to filter CGM readings and timestamps at once 
    for key in test_data_recordings:
        for key2 in test_data_recordings[key]: 
            for key3 in test_data_recordings[key][key2]:
                for key4 in test_data_recordings[key][key2][key3]:
                    for key5 in test_data_recordings[key][key2][key3][key4]:
                        for key6 in test_data_recordings[key][key2][key3][key4][key5]:
                                
                                timestamps = test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["timestamp"]
                                readings = test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["reading"]
                                
                                # Concatenate the timestamp and the readings
                                data = np.column_stack((timestamps, readings))

                                # Add the data to the dictionary
                                test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"] = data

                                #########################################
                                # Filter the data
                                data = data[data[:,0] > subjects_end_training_dates[key]]

                                if verbose == 1: 
                                    print("~~~~")
                                    print(key)
                                    print("Lower limit", subjects_end_training_dates[key])
                                    print("Before filtering", test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["timestamp"][0])
                                    print("After filtering", data[0][0])

                                # Update the dictionary
                                test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"] = data

                                # Difference of last and first timestamps 
                                test_time = test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"][-1][0] - test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"][0][0]
                            
                                if verbose == 1: 
                                    print(key, test_time) 
    
    return test_data_recordings



def extract_test_set_from_test_data_dictionary(test_data_recordings : Dict, NUM_OF_DAYS_TEST_SET : int, verbose : int) -> Dict: 
    
    """
    This function takes the test data dictionary after cleaning the old data belonginh to the training process and the sensor model
    filtering. It outputs the final CGM and timestamps sequences to generate the test vectors. The main parameter is the number of days
    that will form the test set. Subjects that do not have at least this amount of days in their data, will be discared. The more number
    of days, the more subjects will be discarded. 

    Args
    ----   
        test_data_recordings: Dictionary with the test data recordings sorted by ID and sensor.
        NUM_OF_DAYS_TEST_SET: Number of days that will form the test set.
        verbose: Flag to print informative messages.

    Returns: 
    -------
        test_data_recordings: Dictionary with the final sequences of CGM readings and timestamps for the test set of containing NUM_OF_DAYS_TEST_SET
    
    """

    for key in test_data_recordings:
        for key2 in test_data_recordings[key]: 
            for key3 in test_data_recordings[key][key2]:
                for key4 in test_data_recordings[key][key2][key3]:
                    for key5 in test_data_recordings[key][key2][key3][key4]:
                        for key6 in test_data_recordings[key][key2][key3][key4][key5]:
                                
                                # Extract the time data
                                time_data = test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"][:,0]

                                # Calculate the time difference
                                time_diff = time_data[-1] - time_data[0]

                                # Check if the time difference is greater than the threshold
                                if time_diff < NUM_OF_DAYS_TEST_SET:
                                    
                                    # Delete entry 
                                    test_data_recordings[key][key2][key3][key4][key5][key6] = {}
                                
                                else: 

                                    # From the first timestamp, sum the time threshold and only keep that data 
                                    upper_limit = time_data[0] + NUM_OF_DAYS_TEST_SET

                                    # Filter the data
                                    data = test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"]
                                    data = data[data[:,0] < upper_limit]

                                    # Update the dictionary
                                    test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"] = data

                                    # Place the data in CGM readings and timestamps dictionary entries
                                    test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["reading"] = data[:,1]
                                    test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["timestamp"] = data[:,0]

                                    # Delete the concatenated data entry 
                                    del test_data_recordings[key][key2][key3][key4][key5][key6]['CGM']["concat_data"]

    # Save current keys in a list 
    current_keys = list(test_data_recordings.keys())

    # Check if there are IDs completely empty to delete them
    for i in range(0,len(set_of_libreview_keys)): 

        # Avoid errors with IDs already deleted
        if set_of_libreview_keys[i][0] in current_keys: 
            
            # Check if there is data for the subject after temporal filtering 
            for key in test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():
                for key2 in test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key].keys():
                    
                    # Delete entry if there is no data 
                    if len(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]) == 0:
                        
                        # Delete entry 
                        del test_data_recordings[set_of_libreview_keys[i][0]]
                        
                        if verbose == 1: 
                            print(set_of_libreview_keys[i][0], "deleted")    
        else: 
            if verbose == 1: 
                print(set_of_libreview_keys[i][0], "already deleted")
            pass
    
    return test_data_recordings

def final_model_test(test_data_recordings_final : Dict, PH : int, DL_models : List, NUM_OF_DAYS_TEST_SET : int, N : int = 96, step : int = 1) -> None:
    """
    This function performs the final model tests, given the test data recordings, and also the number 
    of days of the test set. The model evaluation is the same as in main_libreview.py. N, step are parameters
    derived from the previous step. Please, note that "libreview_sensors" is hardcoded now, but will be changed 
    soon. 

    Args: 
    ----
        test_data_recording_final : Dictionary containing the filtered test data recordings subject per subject 
        PH : Prediction horizon (in minutes). Currently 30 and 60 have been tested. 
        DL_models: DL models to be tested. (Currently, should be []'LSTM', 'StackedLSTM', 'DIL-1D-UNET'])
        NUM_OF_DAYS_TEST_SET : Number of days of the test set.
        N : Number of input steps. Default is 96. (According to previous steps)
        step : Step between input steps. Default is 1. (According to previous steps)
    
    Returns: 
    -------
        None
    """

    # First, repeat X and Y generation for each subject and then preprocessing 
    # Iterate over the subjects to generate X and Y test vectors
    for key in test_data_recordings_final.keys():
        for key2 in test_data_recordings_final[key]['001']['001']['2024'].keys():
            for key3 in test_data_recordings_final[key]['001']['001']['2024'][key2].keys():

                # Create dictionary to fill it with the results (one per patient) 
                try:
                    with open('test_results_dictionary.json', 'rb') as handle:
                        test_results_dictionary = json.load(handle)
                        print("Dictionary loaded.\n")
                except:
                    test_results_dictionary = {}
                    print("Non-existing dictionary. A new one was created.\n")
    
                # Save CGM and timestamps in a variable 
                recordings = test_data_recordings_final[key]['001']['001']['2024'][key2][key3]['CGM']['reading']
                timestamps = test_data_recordings_final[key]['001']['001']['2024'][key2][key3]['CGM']['timestamp']

                # Save number of available CGM test samples in dictionary 
                test_data_recordings_final[key]['001']['001']['2024'][key2][key3]['CGM']['test_CGM_samples'] = recordings.shape

                # Generate X and Y test 
                X_test, Y_test, X_times, Y_times = get_LibreView_CGM_X_Y_multistep(recordings, timestamps, libreview_sensors, 
                                            N, step, PH, plot = True, verbose = 0) 

                # Min-max normalization
                X_norm = (X_test - np.min(X_test))/(np.max(X_test) - np.min(X_test))
                Y_norm = (Y_test - np.min(X_test))/(np.max(X_test) - np.min(X_test))        

                # Get 1st derivative of X_norm
                X_norm_der = np.diff(X_norm, axis = 1)

                # Add the last point of X_norm_dev on the right of the array tp have same dimension than X_norm
                X_norm_der = np.insert(X_norm_der, -1, X_norm_der[:,-1], axis = 1)

                # Stack X_norm and X_norm_der
                X_norm = np.dstack((X_norm, X_norm_der))

                # Create a key depending on the number of test days 
                test_key = str(NUM_OF_DAYS_TEST_SET)
                test_results_dictionary[test_key] = {}
                
                # Go to the directory where the models are stored. Iterate over the three evaluated 
                for DL_model in DL_models: 
                    
                    # Go to the correspondant directory 
                    dir = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files\{}\N96\step1\PH{}\multi\month-wise-4-folds\norm_min-max\None_sampling\{}\ISO_loss\1-yr_model\training".format(key, PH, DL_model)
                    os.chdir(dir)

                    # Load the model 
                    name = "1yr-"+DL_model
                    model = tf.keras.models.load_model(name+'.h5', custom_objects={'ISO_adapted_loss': ISO_adapted_loss})

                    # Model evaluation 
                    results_normal_eval = multi_step_model_evaluation(N, PH, name, "min-max", 2, X_norm, Y_norm, round(PH/15), X_test, "ISO_loss", plot_results=True)

                    # Create a key depending on the number of test days 
                    test_key = str(NUM_OF_DAYS_TEST_SET)
                    
                    # Save results in directory 
                    test_results_dictionary[test_key][DL_model] = results_normal_eval

                # Save updated dictionary 
                with open('test_results_dictionary.json', 'w') as fp:
                    json.dump(test_results_dictionary, fp) 


def subject_per_subject_bar_diagram(test_data_recordings_final : Dict, metric : str, metric_LSTM : List, metric_StackedLSTM : List, metric_DIL_1D_UNET, PH : int, NUM_OF_DAYS_TEST_SET : int) -> None: 
    """
    Given the dictionary with the IDs of the included subjects and the name of the metric
    evaluated, this function generates a bar diagram with the metric evaluated for each subject.

    Args: 
    -----
        test_data_recordings : Dictionary with the test data recordings
        metric : Name of the metric evaluated
        metric_LSTM : List with the metric values for LSTM
        metric_StackedLSTM : List with the metric values for Stacked LSTM
        metric_DIL_1D_UNET : List with the metric values for DIL-1D-UNET
        PH : Prediction horizon (in minutes). Currently 30 and 60 have been tested.
        NUM_OF_DAYS_TEST_SET : Number of days included in the test set

    Returns:
    --------
        None
    """

    # Plot all values of RMSE_LSTM in a bar diagram 

    # Patient list (filled manually)
    all_patient_list_sorted = ['004', '011', '029', '008', '015', '045', '025', '065', '067', '026', '060',
                        '062', '039', '007', '048', '001', '014', '013', '046', '043', '051', '049',
                        '063', '055', '061', '057', '003', '068', '058']

    # Filter all patients with no test instances
    filtered_patient_list = [x for x in all_patient_list_sorted if x in list(test_data_recordings_final.keys())]

    filtered_patient_list

    plt.figure(figsize=(17, 8.5))

    # Set font to arial 
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Set text to bold
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    # Grouped bar diagram for RMSE in LSTM, Stacked LSTM and DIL-1D-UNET
    barWidth = 0.25

    bars1 = metric_LSTM
    bars2 = metric_StackedLSTM
    bars3 = metric_DIL_1D_UNET

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, bars1, color='b', width=barWidth, edgecolor='grey', label='LSTM')
    plt.bar(r2, bars2, color='r', width=barWidth, edgecolor='grey', label='Stacked LSTM')
    plt.bar(r3, bars3, color='g', width=barWidth, edgecolor='grey', label='DIL-1D-UNET')

    # X labels are the filtered patient_list
    plt.xticks(range(len(filtered_patient_list)), filtered_patient_list)

    # Center the x ticks
    plt.xlabel('Subject', fontweight='bold')

    plt.legend()

    if metric == 'RMSE': 
        plt.ylabel('RMSE (mg/dL)', fontweight='bold')
    elif metric == 'PARKES':
        plt.ylabel('PARKES (%)', fontweight='bold')

        # Add a dash line at 99% of the PARKES error
        plt.axhline(y=99, color='black', linestyle='--', label='99% PARKES error')

    elif metric == 'ISO':
        plt.ylabel('ISO (%)', fontweight='bold')

        # Add a dash line at 99% of the PARKES error  
        plt.axhline(y=95, color='black', linestyle='--', label='99% PARKES error')

    # Save the figure 
    plt.savefig(metric+'PH-'+str(PH)+'min_'+str(NUM_OF_DAYS_TEST_SET)+ '_test.svg', dpi=1200)

    plt.show()

def final_DIY_models_test(data_dict : Dict, sensor_black_list : List, PH : int, NUM_OF_DAYS_TEST_SET : int,
                        N : int = 96, step : int = 1,
                        DL_models : List = ['LSTM', 'StackedLSTM', 'DIL-1D-UNET'],
                        parent_dir : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files",
                        TEST_DATASET_PATH : str = r"C:\Users\aralmeida\Downloads\Datos sensor FSL 2024\Datos crudos excel 2024_adapted") ->  None: 
    
    """
    This function performs the final tests with the DIY models trained within 1 year. In this function, data from the previous 
    cross-validation step is used to filter the test data ny taking the sensor MACs. Thus, if a subject has changed his/her
    sensor and no new data is provided, it will be discarded. The test data is also filtered by the sensor model.
    Besides, if interruptions of lack in data samples imply that no test intances are generated, this subject will be also discarded.
    Notice that the N, PH and NUM_OF_DAYS_TEST_SET will influence on this. The final test set is extracted from the test data dictionary.
    Once the data is extracted, the evaluation is done as in main_libreview.py.

    For more details about the specific steps, please refer to the specific functions inside this one. 

    Args:
    ----
        data_dict: Dictionary with the CGM data after read the .csv files. 
        sensor_black_list: List of sensors to be discarded from the dictionary.
        PH : Prediction horizon (in minutes). Currently 30 and 60 have been tested. 
        NUM_OF_DAYS_TEST_SET : Number of days of the test set. 
        N : Number of input steps. Default is 96. (According to previous steps)
        step : Step between input steps. Default is 1. (According to previous steps)
        DL_models: DL models to be tested. Deafult: ['LSTM', 'StackedLSTM', 'DIL-1D-UNET']
        parent_dir : Parent directory where the data is stored. Set by default but should be changed if someone wants to use it. 
        TEST_DATASET_PATH : Path to the test dataset. Set by default but should be changed if someone wants to use it.
    
    Returns: 
    -------
        test_data_recordings: Dictionary with the final sequences of CGM readings and timestamps for the test set of containing NUM_OF_DAYS_TEST_SET. 

    """

    # Get the end dates of the training set 
    subjects_end_training_dates = get_end_training_dates(parent_dir, TEST_DATASET_PATH)

    # Extract the test data recordings 
    test_data_recordings = extract_test_data_recordings(subjects_end_training_dates, data_dict)

    # Filter subjects using sensor from the blacklist 
    test_data_recordings = discard_data_from_sensor_black_list(test_data_recordings, sensor_black_list)

    # Open the dicionary previously generated with the training and validation data 
    os.chdir(parent_dir)

    # Go to parent folder
    os.chdir("..")

    # Open libreview_1_yr_recordings pickle
    with open('libreview_data_1yr_recordings.pickle', 'rb') as handle:
        yr_data = pickle.load(handle)

    # Back to the test set directory 
    os.chdir(TEST_DATASET_PATH)

    # Extract the MAC of the sensors from the subjects used in the previous step (4-folds cross validation)
    data_with_1_yr_sensor_MAC = get_ID_sensor_MAC_pairs(yr_data)

    # Inclusion criteria #1: Subjects that have the same sensor during the 1-year period to train the models
    # Filter subjects that change their sensors to exclude then from the final test
    test_data_recordings = filter_subjects_that_change_sensor(test_data_recordings, data_with_1_yr_sensor_MAC)

    # Check if there are IDs completely empty to delete them
    for i in range(0,len(set_of_libreview_keys)):      
        if len(test_data_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]]) == 0:

            # Delete entry 
            del test_data_recordings[set_of_libreview_keys[i][0]]

    print("After filtering sensor changes: ", len(test_data_recordings.keys()))

    # Get the IDs MACs peers for the test data 
    data_test_sensor_MAC = get_ID_sensor_MAC_pairs(test_data_recordings)

    # Remove, if exists, the overlapping data to test only with data not used in the training and validation sets
    test_data_recordings = remove_training_data_from_test_set(test_data_recordings, subjects_end_training_dates, 1)

    # Establishing the period of the test set (30, 90, 180, and 365 days in this work), extract the final test set
    test_data_recordings = extract_test_set_from_test_data_dictionary(test_data_recordings, NUM_OF_DAYS_TEST_SET, 1)

    # List to store the keys of the subjects with no test instances 
    no_test_subjects = []

    # Generate X and Y to see if there are subjects that do not provide any instances and delete them 
    for key in test_data_recordings.keys():
        for key2 in test_data_recordings[key]['001']['001']['2024'].keys():
            for key3 in test_data_recordings[key]['001']['001']['2024'][key2].keys():

                # # Create dictionary to fill it with the results (one per patient) 
                # test_results_dictionary = create_LibreView_results_dictionary()
    
                # Save CGM and timestamps in a variable 
                recordings = test_data_recordings[key]['001']['001']['2024'][key2][key3]['CGM']['reading']
                timestamps = test_data_recordings[key]['001']['001']['2024'][key2][key3]['CGM']['timestamp']

                # Generate X and Y test 
                X_test, Y_test, X_times, Y_times = get_LibreView_CGM_X_Y_multistep(recordings, timestamps, libreview_sensors, 
                                            N, step, PH, plot = True, verbose = 0) 

                # Save number of available CGM test samples in dictionary 
                test_data_recordings[key]['001']['001']['2024'][key2][key3]['CGM']['test_CGM_instances'] = X_test.shape[0]

    for key in test_data_recordings.keys():
        for key2 in test_data_recordings[key]['001']['001']['2024'].keys():
            for key3 in test_data_recordings[key]['001']['001']['2024'][key2].keys():

                if test_data_recordings[key]['001']['001']['2024'][key2][key3]['CGM']['test_CGM_instances'] == 0:

                    no_test_subjects.append(key)

    # Delete entry with no test instances
    for i in range(0,len(no_test_subjects)):      
        del test_data_recordings[no_test_subjects[i]]

    print("After filtering because subject does not have test instances: ", len(test_data_recordings.keys()))

    final_model_test(test_data_recordings, PH, DL_models, NUM_OF_DAYS_TEST_SET)

    # Save the final test data recordings
    with open('test_final_data_recordings.pickle', 'wb') as handle:
        pickle.dump(test_data_recordings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return test_data_recordings


def group_and_save_metrics(test_data_recordings_final : Dict, DL_models : List, PH : int, NUM_OF_DAYS_TEST_SET : int) -> List:
    
    """
    Group RMSE, Parkes and ISO metrics for LSTM, Stacked LSTM, DIL-1D-UNET and Naive models. 
    Once this function is called, the results are saved in an Excel file an ready to be 
    plotted through the use of Lists.  

    Args:
    ----
        test_data_recordings_final : Dictionary containing the filtered test data recordings subject per subject 
        DL_models : List with the DL models to be tested. 
        PH : Prediction horizon (in minutes). Currently 30 and 60 have been tested.
        NUM_OF_DAYS_TEST_SET : Number of days included in the test set.

    Returns:
    -------
        RMSE_LSTM : List with the RMSE values for LSTM
        RMSE_StackedLSTM : List with the RMSE values for Stacked LSTM
        RMSE_DIL_1D_UNET : List with the RMSE values for DIL-1D-UNET
        RMSE_naive : List with the RMSE values for Naive model
        PARKES_LSTM : List with the PARKES values for LSTM
        PARKES_StackedLSTM : List with the PARKES values for Stacked LSTM
        PARKES_DIL_1D_UNET : List with the PARKES values for DIL-1D-UNET
        PARKES_naive : List with the PARKES values for Naive model
        ISO_LSTM : List with the ISO values for LSTM
        ISO_StackedLSTM : List with the ISO values for Stacked LSTM
        ISO_DIL_1D_UNET : List with the ISO values for DIL-1D-UNET
        ISO_naive : List with the ISO values for Naive model   
    """

    # Group the RMSE of all patients
    RMSE_LSTM = []
    RMSE_StackedLSTM = []
    RMSE_DIL_1D_UNET = []
    RMSE_naive = []

    PARKES_LSTM = []
    PARKES_StackedLSTM = []
    PARKES_DIL_1D_UNET = []
    PARKES_naive = []

    ISO_LSTM = []
    ISO_StackedLSTM = []
    ISO_DIL_1D_UNET = []
    ISO_naive = []

    for key in test_data_recordings_final.keys():
                
                # Go to the correspondant directory 
                dir = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files\{}\N96\step1\PH{}\multi\month-wise-4-folds\norm_min-max\None_sampling\{}\ISO_loss\1-yr_model\evaluation".format(key, PH, DL_models[2])
                os.chdir(dir)

                # Load dictionary
                with open('test_results_dictionary.json', 'r') as fp:
                    curr_results = json.load(fp)

                # Index depending on the PH 
                if PH == 30:
                    idx = 1
                if PH == 60:
                    idx = 3
                
                # Append RMSE, Parkes, ISO
                RMSE_LSTM.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[0]]['RMSE'][idx])
                PARKES_LSTM.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[0]]['PARKES'][idx])
                ISO_LSTM.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[0]]['ISO'][idx])

                # Go to the correspondant directory 
                dir = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files\{}\N96\step1\PH{}\multi\month-wise-4-folds\norm_min-max\None_sampling\{}\ISO_loss\1-yr_model\evaluation".format(key, PH, DL_models[2])
                os.chdir(dir)

                # Load dictionary
                with open('test_results_dictionary.json', 'r') as fp:
                    curr_results = json.load(fp)

                RMSE_StackedLSTM.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[1]]['RMSE'][idx])
                PARKES_StackedLSTM.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[1]]['PARKES'][idx])
                ISO_StackedLSTM.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[1]]['ISO'][idx])

                # Go to the correspondant directory 
                dir = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files\{}\N96\step1\PH{}\multi\month-wise-4-folds\norm_min-max\None_sampling\{}\ISO_loss\1-yr_model\evaluation".format(key, PH, DL_models[2])
                os.chdir(dir)

                # Load dictionary
                with open('test_results_dictionary.json', 'r') as fp:
                    curr_results = json.load(fp)

                RMSE_DIL_1D_UNET.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[2]]['RMSE'][idx])
                PARKES_DIL_1D_UNET.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[2]]['PARKES'][idx])
                ISO_DIL_1D_UNET.append(curr_results[str(NUM_OF_DAYS_TEST_SET)][DL_models[2]]['ISO'][idx])

    # Calculate the mean RMSE for each model
    mean_RMSE_LSTM = np.mean(RMSE_LSTM)
    mean_RMSE_StackedLSTM = np.mean(RMSE_StackedLSTM)
    mean_RMSE_DIL_1D_UNET = np.mean(RMSE_DIL_1D_UNET)

    # Calculate the standard deviation of the RMSE for each model
    std_RMSE_LSTM = np.std(RMSE_LSTM)
    std_RMSE_StackedLSTM = np.std(RMSE_StackedLSTM)
    std_RMSE_DIL_1D_UNET = np.std(RMSE_DIL_1D_UNET)

    # Print mean and std of the metrics for each model 
    print("\nRMSE LSTM: ", mean_RMSE_LSTM, "",  std_RMSE_LSTM)
    print("RMSE Stacked LSTM: ", mean_RMSE_StackedLSTM, "", std_RMSE_StackedLSTM)
    print("RMSE DIL-1D-UNET: ", mean_RMSE_DIL_1D_UNET, "", std_RMSE_DIL_1D_UNET)

    # Calculate the mean Parkes for each model
    mean_PARKES_LSTM = np.mean(PARKES_LSTM)
    mean_PARKES_StackedLSTM = np.mean(PARKES_StackedLSTM)
    mean_PARKES_DIL_1D_UNET = np.mean(PARKES_DIL_1D_UNET)

    # Calculate the standard deviation of the Parkes for each model
    std_PARKES_LSTM = np.std(PARKES_LSTM)
    std_PARKES_StackedLSTM = np.std(PARKES_StackedLSTM)
    std_PARKES_DIL_1D_UNET = np.std(PARKES_DIL_1D_UNET)

    # Print mean and std of the metrics for each model
    print("\nPARKES LSTM: ", mean_PARKES_LSTM, "",  std_PARKES_LSTM)
    print("PARKES Stacked LSTM: ", mean_PARKES_StackedLSTM, "", std_PARKES_StackedLSTM)
    print("PARKES DIL-1D-UNET: ", mean_PARKES_DIL_1D_UNET, "", std_PARKES_DIL_1D_UNET)

    # Calculate the mean ISO for each model
    mean_ISO_LSTM = np.mean(ISO_LSTM)
    mean_ISO_StackedLSTM = np.mean(ISO_StackedLSTM)
    mean_ISO_DIL_1D_UNET = np.mean(ISO_DIL_1D_UNET)

    # Calculate the standard deviation of the ISO for each model
    std_ISO_LSTM = np.std(ISO_LSTM)
    std_ISO_StackedLSTM = np.std(ISO_StackedLSTM)
    std_ISO_DIL_1D_UNET = np.std(ISO_DIL_1D_UNET)

    # Print mean and std of the metrics for each model
    print("\nISO LSTM: ", mean_ISO_LSTM, "",  std_ISO_LSTM)
    print("ISO Stacked LSTM: ", mean_ISO_StackedLSTM, "", std_ISO_StackedLSTM)
    print("ISO DIL-1D-UNET: ", mean_ISO_DIL_1D_UNET, "", std_ISO_DIL_1D_UNET)

    return RMSE_LSTM, RMSE_StackedLSTM, RMSE_DIL_1D_UNET, PARKES_LSTM, PARKES_StackedLSTM, PARKES_DIL_1D_UNET, ISO_LSTM, ISO_StackedLSTM, ISO_DIL_1D_UNET
