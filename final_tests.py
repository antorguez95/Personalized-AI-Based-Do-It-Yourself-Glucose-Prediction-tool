from typing import Dict, List 
import os
import pickle
import numpy as np 
import pandas as pd

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