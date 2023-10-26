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


from typing import Dict 
import pickle
import os
import pandas as pd
import numpy as np 
import json 


# Set of keys of all patients (41) to easily access their data in the Libreview files (more .csv files would imply changing this)
set_of_libreview_keys = [["001", "001", "001", "12-6-2023"],
            ["003", "001", "001", "12-6-2023"],
            ["004", "001", "001", "10-7-2023"],
            ["007", "001", "001", "10-7-2023"], 
            ["008", "001", "001", "10-7-2023"],
            ["011", "001", "001", "10-7-2023"],
            ["013", "001", "001", "10-7-2023"],
            ["014", "001", "001", "10-7-2023"],
            ["015", "001", "001", "11-7-2023"],
            ["019", "001", "001", "11-7-2023"],
            ["020", "001", "001", "4-9-2023"],
            ["025", "001", "001", "11-7-2023"],
            ["026", "001", "001", "11-7-2023"],
            ["029", "001", "001", "11-7-2023"],
            ["039", "001", "001", "11-7-2023"],
            ["042", "001", "001", "11-7-2023"],
            ["043", "001", "001", "11-7-2023"],
            ["044", "001", "001", "11-7-2023"],
            ["045", "001", "001", "11-7-2023"],
            ["046", "001", "001", "11-7-2023"],
            ["047", "001", "001", "11-7-2023"],
            ["048", "001", "001", "11-7-2023"],
            ["049", "001", "001", "11-7-2023"],
            ["051", "001", "001", "11-7-2023"],
            ["052", "001", "001", "4-9-2023"],
            ["053", "001", "001", "4-9-2023"],
            ["054", "001", "001", "4-9-2023"],
            ["055", "001", "001", "4-9-2023"],
            ["056", "001", "001", "4-9-2023"],
            ["057", "001", "001", "4-9-2023"],
            ["058", "001", "001", "4-9-2023"],
            ["059", "001", "001", "4-9-2023"],
            ["060", "001", "001", "4-9-2023"],
            ["061", "001", "001", "4-9-2023"],
            ["062", "001", "001", "4-9-2023"],
            ["063", "001", "001", "4-9-2023"],
            ["064", "001", "001", "4-9-2023"],
            ["065", "001", "001", "6-9-2023"],
            ["066", "001", "001", "6-9-2023"],
            ["067", "001", "001", "6-9-2023"],
            ["068", "001", "001", "6-9-2023"]]

def prepare_LibreView_data(dataset_path : str, save_dict : bool = True) -> Dict:
    
    """
    Function to prepare the data stored in .csv from LibreView application
    for its further processing. Current sensors supported are:
        - FreeStyle LibreLink 
            - Glucose
            - Insulin
            etc
        - FreeStyle Libre 3
            - Glucose
            - Insulin
            etc
    
    This function returns a dictionary with the following structure:
    data_dict = {id : {s : {r : {download_date : {MAC : {sensor_name : {variable : {reading : np.array, timestamp : np.array}}}}}}}}
    where:
        - id : patient id
        - s : sensor id 
        - r : recording id 
        - download_date : date of the download 
        - MAC : MAC address of the sensor 
        - sensor_name : name of the sensor (within the ones described above) 
        - variable : type of the reading (e.g., CGM, fast insulin, etc) 
        - reading : sensor reading 
        - timestamp : sensor reading timestamp
    
    It saves the dictionary in a .pickle is the flag is set to True, and it saves an Exccel file (.xslx) with the summary of 
    the read data. 
    
    Args
    ----
        dataset_path : Path to the dataset directory
        save_json : Flag to save the dictionary in a .json file. Default is True.

    Returns
    -------
        data_dict : Dictionary with the structure described above

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
                current_recordings = pd.read_csv(file, low_memory=False)

                # Clean NaN values
                current_recordings = current_recordings.dropna(axis=0, subset=['Tipo de registro'])

                # Recording #14-01-01 has an error in the timestamps from sample 71870 to sample 74580. These are removed
                if id == "014" and s == "001" and r == "001" : 
                    idxs = np.where(current_recordings['Sello de tiempo del dispositivo'] == '01-01-0001 00:00')
                    current_recordings.drop(current_recordings.index[71870:74581], inplace=True)

                # Conver timestamps to datetime64
                current_recordings['Sello de tiempo del dispositivo'] = pd.to_datetime(current_recordings['Sello de tiempo del dispositivo'],
                                                                                    dayfirst=True,
                                                                                    format="%d-%m-%Y %H:%M",
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
                        data_dict[id][s][r][download_date][MACs[i]] = {sensor_name : {"CGM" : {"reading" : np.empty((0)),
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
        filename = 'libreview_data.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dict

def generate_LibreView_npy_files(libreview_data : Dict, foldername : str = r"\npy_files", verbose : int = 0) -> None: 
    
    """
    From a dictionary created after reading the Libreview .csv files, this function generates
    generates one .npy file per patient, sensor and recording in the '/npy_files' .It generates
    one folder per ID, and one .npy file per sensor and recording. The reason of making two functions
    is that the .csv reading is time consuming, so it is better to generate the .npy files from the 
    saved dictionary generated with prepare_LibreView_data(). 

    Args
    ----
        libreview_data : Dictionary generated with prepare_LibreView_data()
        foldername : Name of the folder where the .npy files will be stored. Default: r"\npy_files"
        verbose : grade of verbosity. Default: 0. 
    
    Returns
    -------
        None      
    """

    # Create Pandas DataFrame with the data 
    summary_df = pd.DataFrame.from_dict(libreview_data, columns=["Patient ID", "S", "R", "Nº of CGM samples", "Data from 1st sample",
                                                                "Data from last sample", "Days bewteen readings", "Sensor"], orient="index")

    # Counter to be capable of index the same ID with every sensor
    id_MAC_combination = 0

    # Iterate over all dictionary keys
    for i in range(0,len(set_of_libreview_keys)):

        # The try except is to avoid the KeyError when there is no data for a given sensor
        try: 
            for key in libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():
                for key2 in libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key].keys():
                    
                    
                    if verbose == 1 : 
                        print("Samples of CGM in patient #",set_of_libreview_keys[i][0], " in sensor", key2,": ", libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"].shape[0])
                    
                    summary_df.loc[id_MAC_combination, "Patient ID"] = set_of_libreview_keys[i][0]
                    summary_df.loc[id_MAC_combination, "S"] = set_of_libreview_keys[i][1]
                    summary_df.loc[id_MAC_combination, "R"] = set_of_libreview_keys[i][2]
                    summary_df.loc[id_MAC_combination, "Sensor"] = key2

                    if not libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"].any():
                        
                        if verbose == 1: 
                            print("No CGM data for patient #",set_of_libreview_keys[i][0], " in sensor", key2)
                        summary_df.loc[id_MAC_combination, "Data from 1st sample"] = "No CGM data"
                        summary_df.loc[id_MAC_combination, "Data from last sample"] = "No CGM data"
                        summary_df.loc[id_MAC_combination, "Nº of CGM samples"] = "No CGM data"
                    
                    else: 

                        summary_df.loc[id_MAC_combination, "Data from 1st sample"] = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"][0]
                        summary_df.loc[id_MAC_combination, "Data from last sample"] = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"][-1]
                        summary_df.loc[id_MAC_combination, "Days bewteen readings"] = summary_df.loc[id_MAC_combination, "Data from last sample"] - summary_df.loc[id_MAC_combination, "Data from 1st sample"]
                        summary_df.loc[id_MAC_combination, "Nº of CGM samples"] = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"].shape[0]
                    
                    # Update counter
                    id_MAC_combination += 1
        except: 
            pass
        
    # Export summary DataFrame to Excel file 
    summary_df.to_excel("Libreview_Patients_CGM_Data_Summary.xlsx")

    # Save current directory 
    dir = os.getcwd()

    # Create directory if it has not been previously created
    if foldername not in dir:  
        # Create directory to store the training parameters and results
        path = foldername
        if not os.path.exists(dir+path):
            os.makedirs(dir+path)
        # Change to that directory 
        os.chdir(dir+path)

    # Go to directory 
    os.chdir(dir+foldername)

    # Counter to count the number of different sensors that a patient has used 
    num_of_ID_reading = 1

    # Export the CGM data of each separate ID CGM reading as npy file
    for i in range(0,len(set_of_libreview_keys)):

        # Create a new folder for each patient if not previously created 
        if set_of_libreview_keys[i][0] not in os.listdir():

            # Create folder only if the current directory contains the ID, or it is a subset of that dictionary 
            if set_of_libreview_keys[i][0] in libreview_data.keys(): 
                os.mkdir(set_of_libreview_keys[i][0])

                # Change to that directory
                os.chdir(set_of_libreview_keys[i][0])
         
        # Same as before: try except to avoid the KeyError when there is no data for a given sensor
        try :

            num_of_ID_reading = 1

            for key in libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():
                for key2 in libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key].keys():
                    filename = "X_" + set_of_libreview_keys[i][0] + "_" + str(num_of_ID_reading) + "_" + key2 + "_CGM.npy"
                    timestamp_filename = "X_" + set_of_libreview_keys[i][0] + "_" + str(num_of_ID_reading) + "_" + key2 + "_CGM_timestamp.npy"
                    
                    # Save readings and timestamps as npy files
                    np.save(filename, libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"])
                    np.save(timestamp_filename, libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"])

                    # Increment counter
                    num_of_ID_reading += 1
        except: 
            pass
            
        # Go to parent directory just in case the directory was created 
        if set_of_libreview_keys[i][0] not in os.listdir():
            if set_of_libreview_keys[i][0] in libreview_data.keys(): 
                os.chdir("..")

def get_1year_LibreView_recordings_dict(libreview_data : Dict) -> Dict: 

    """
    This function takes, from a dictionary generated from the raw .csv files,
    the dictionary entries that have at least one year of data in a raw with the same sensor.
    It returns a dictionary with the same structure as the input dictionary, but filtered. 

    Args
    ----
        libreview_data : Dictionary generated with prepare_LibreView_data()
    
    Returns
    -------
        data_1yr_recordings : Dictionary with the same structure as the input dictionary only 
        with the recordings with at least one year in a raw with the same sensor. 
    """


    # Subset of the original dictionary with the valid recordings (duration >= 1 year)
    data_1yr_recordings = {}

    # Iterate over all dictionary keys
    for i in range(0,len(set_of_libreview_keys)):

        # Initialize the dictionary entries
        data_1yr_recordings[set_of_libreview_keys[i][0]] = {}
        data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]] = {}
        data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]] = {}
        data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]] = {}

        for key in libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys():

            # Initialize the entry regarding the MAC
            data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key] = {}

            for key2 in libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key].keys():

                # There are entries without data (e.g., 014)
                if not libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"].any():
                    pass
                else: 

                    # Calculate number of days so we can discard recordings with less than a year of data
                    data_1st_sample = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"][0]
                    data_last_sample = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"][-1]
                    time_between_readings = data_last_sample - data_1st_sample

                    if time_between_readings.days >= 365:

                        # Corresponding CGM readings and their timestamps
                        cgm = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["reading"]
                        cgm_timestamp = libreview_data[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2]["CGM"]["timestamp"]

                        
                        # Fill dictionary with readings of at least one year
                        data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key][key2] = {"CGM" : {"reading" : cgm,
                                                                                                                                            "timestamp" : cgm_timestamp}}

    # Iterate over all dictionary keys to delete the entries that are empty (meaning that they had <1 year of data )
    for i in range(0,len(set_of_libreview_keys)):      
        for key in list(data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]].keys()):
        
                # Check if the entry is empty or not to delete it 
                if len(data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]) == 0:

                    # Delete entry 
                    del data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]][key]

    # Check if there are IDs completely empty to delete them
    for i in range(0,len(set_of_libreview_keys)):      
        if len(data_1yr_recordings[set_of_libreview_keys[i][0]][set_of_libreview_keys[i][1]][set_of_libreview_keys[i][2]][set_of_libreview_keys[i][3]]) == 0:

            # Delete entry 
            del data_1yr_recordings[set_of_libreview_keys[i][0]]

    # Save dictionary as pickle
    with open('libreview_data_1yr_recordings.pickle', 'wb') as handle:
        pickle.dump(data_1yr_recordings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print the number of patients that will be used to develope the prediction models 
    print("Number of patients with at least one year of CGM data: ", len(data_1yr_recordings.keys()))

    return data_1yr_recordings

def generate_LibreView_npy_1yr_recordings(data_1yr_recordings : Dict): 
    
    """
    This functions extract, from a dictionary containing the recordings 
    having at least 1 year of consecutive CGM readings, the data and timestamps 
    of the oldest recording of each ID (in case the ID has more then one recording
    with >= year). The .npy that are read were previously generated by 
    generate_LibreView_npy_files() function.  From this recordings, an exact one
    year is extracted to train and test the models with an exact 1 year.
    These recordings and timestamps are saved as .npy files in the ID correspondant
    directory. 

    Args 
    ----
        data_1yr_recordings : dictionary containing all the entries that has >= year of CGM data

    Returns
    -------
        None 
    """
    
    # Iterate over keys to check how many entries are there about each patient 
    for id in data_1yr_recordings.keys():
        for s in data_1yr_recordings[id].keys(): # This has been validated when S and R are one per ID
            for r in data_1yr_recordings[id][s].keys():
                for data in data_1yr_recordings[id][s][r].keys():

                    if len(data_1yr_recordings[id][s][r][data]) == 1: 
                        
                        # Get current MAC and sensor name 
                        mac = list(data_1yr_recordings[id][s][r][data].keys())[0]
                        sensor = list(data_1yr_recordings[id][s][r][data][mac].keys())[0]

                        # Go to the directory where the .npy files of the current patient are stored
                        os.chdir(id)

                        # Set index to 1 because there are not more recordings for the current patient 
                        older_recording_idx = '1' 

                        # Read .npy files
                        recording = np.load('X_{}_{}_{}_CGM.npy'.format(id, older_recording_idx, sensor))
                        recording_timestamps = np.load('X_{}_{}_{}_CGM_timestamp.npy'.format(id, older_recording_idx, sensor), allow_pickle=True)

                        # Take only the first year of data 
                        first_sample = recording_timestamps[0]
                        last_sample_1yr = first_sample + pd.Timedelta(days=365)
                        recording_1yr = recording[recording_timestamps >= first_sample]
                        recording_1yr = recording_1yr[recording_timestamps <= last_sample_1yr]
                        recording_timestamps_1yr = recording_timestamps[recording_timestamps >= first_sample]
                        recording_timestamps_1yr = recording_timestamps_1yr[recording_timestamps <= last_sample_1yr]

                        # Save as .npy file 
                        np.save('oldest_1yr_CGM.npy', recording_1yr)
                        np.save('oldest_1yr_CGM_timestamp.npy', recording_timestamps_1yr)

                        # Back to parent directory 
                        os.chdir('..')
    
                    else: 

                        # Counter
                        iter = 0

                        for mac in data_1yr_recordings[id][s][r][data].keys():
            
                            # Each MAC is uniquely joint to a sensor
                            sensor = list(data_1yr_recordings[id][s][r][data][mac].keys())[0]

                            # Check if the samples are more than 20.000 to considere it as a valid recording
                            if data_1yr_recordings[id][s][r][data][mac][sensor]['CGM']['reading'].shape[0] >= 20000:

                                # Refresh current MAC-associated date
                                first_date_current_mac = data_1yr_recordings[id][s][r][data][mac][sensor]['CGM']['timestamp'][0]

                                if iter == 0:
                                    oldest_mac = mac
                                    oldest_mac_first_date = first_date_current_mac
                                    iter = iter+1
                                
                                else: 
                                    if (first_date_current_mac <= oldest_mac_first_date):
                                        
                                        # Sensor associated to the oldest MAC
                                        sensor = list(data_1yr_recordings[id][s][r][data][oldest_mac].keys())[0]
                                        
                                        # Check if the samples are more than 20.000 to considere it as a valid recording
                                        if data_1yr_recordings[id][s][r][data][oldest_mac][sensor]['CGM']['reading'].shape[0] >= 20000:
                                            
                                            # Update oldest MAC and its first date
                                            oldest_mac = mac
                                            oldest_mac_first_date = first_date_current_mac
                                            iter = iter+1
                                        else: 
                                            pass # the previous oldest MAC is still considered as the oldest one
                                    else: 
                                        # MAC remains the same as before
                                        oldest_mac = oldest_mac 
                                        oldest_mac_first_date = oldest_mac_first_date
                                        iter = iter+1
                            else: # Nothing happens if the recording is not long enough 
                                pass

                        # Sensor associated to the oldest MAC
                        sensor = list(data_1yr_recordings[id][s][r][data][oldest_mac].keys())[0]

                        # Extract CGM recordings of the oldest MAC and their correspondant timestamps 
                        recording = data_1yr_recordings[id][s][r][data][oldest_mac][sensor]['CGM']['reading']
                        recording_timestamps = data_1yr_recordings[id][s][r][data][oldest_mac][sensor]['CGM']['timestamp']

                        # Take only the first year of data 
                        first_sample = recording_timestamps[0]
                        last_sample_1yr = first_sample + pd.Timedelta(days=365)
                        recording_1yr = recording[recording_timestamps >= first_sample]
                        recording_1yr = recording_1yr[recording_timestamps <= last_sample_1yr]
                        recording_timestamps_1yr = recording_timestamps[recording_timestamps >= first_sample]
                        recording_timestamps_1yr = recording_timestamps_1yr[recording_timestamps <= last_sample_1yr]
                                
                        # Save as .npy files
                        os.chdir(id)
                        np.save('oldest_1yr_CGM.npy', recording_1yr)
                        np.save('oldest_1yr_CGM_timestamp.npy', recording_timestamps_1yr)

                        # Back to parent directory
                        os.chdir('..')

def get_oldest_year_npys_from_LibreView_csv(dataset_path : str): 
    """
    From the raw .csv files obtained from LibreView, this function 
    generates numpy files of the oldest year of CGM data of each patient
    without interruptions. Patients that do not have at least one year 
    of data are not considered. For more information about how the data is extrected,
    please refer to the documentation of every particular function. Files
    are stored in the '/1yr_npy_files' folder. CGM recordings are stored as
    'oldest_1yr_CGM.npy' and their timestamps as 'oldest_1yr_CGM_timestamp.npy' 

    Args
    ----
        dataset_path : path where the .csv files are stored.
    
    Returns
    -------
        None 
    """
    # Go to the dataset directory 
    os.chdir(dataset_path)

    # Read .csv or load the pickle file that contains the dictionary to avoid .csv slower reading 
    if 'libreview_data.pickle' in os.listdir():
        with open('libreview_data.pickle', 'rb') as handle:
            libreview_data = pickle.load(handle) # Previously generated with prepare_LibreView_data(DATASET_PATH)
    else: 
        # If the dictionary has not been created, read the .csv files
        libreview_data = prepare_LibreView_data(dataset_path)


    # Take only the T1DM patients with at least one year in a row of CGM data with the same sensor 
    data_1yr_recordings = get_1year_LibreView_recordings_dict(libreview_data)

    # Generate the Libreview .npy files from the generated or saved dictionary 
    generate_LibreView_npy_files(data_1yr_recordings, r"/1yr_npy_files")

    # Extract an EXACT 1 year recordings from the dictionary and store them to load them separately 
    generate_LibreView_npy_1yr_recordings(data_1yr_recordings)

def create_LibreView_results_dictionary(): 
    """
    The results dictionary is updated or created to store the results of the
    experiments. The format to save and load the dictionary is json.

    Args:
    ----
        None
    
    Returns:
    -------
        results_dictionary (dict): Dictionary to store the results of the experiments.

    """ 

    # Read the results from dictionary. If not, create one
    try:
        with open('results_dictionary.json', 'rb') as handle:
            results_dictionary = json.load(handle)
            print("Dictionary loaded.\n")
    except:
        results_dictionary = {}
        print("Non-existing dictionary. A new one was created.\n")

        # # Save dictionary as json
        # with open('results_dictionary.json', 'w') as fp:
        #     json.dump(results_dictionary, fp)  

    return results_dictionary
