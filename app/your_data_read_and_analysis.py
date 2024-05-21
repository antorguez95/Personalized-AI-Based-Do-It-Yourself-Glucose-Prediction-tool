import os 
import pickle
import numpy as np 
import pandas as pd
from typing import Dict, List 

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

def prepare_LibreView_data(your_data_path : str, save_dict : bool = False) -> Dict:
    
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
        save_dict : Flag to save the dictionary in a .pickle file. Default is False.

    Returns
    -------
        data_dict : Dictionary with the structure described above

    """
    
    # Go to the dataset directory 
    os.chdir(your_data_path)
    
    # Create empty dictionary 
    data_dict = {}

    # First time, so the only file is placed is the one uploaded. Error message arise if ID is not in the filename
    for filename in os.listdir(your_data_path) : 

        # Only iterate on the .csv files that contains patient's data 
        if "pickle" in filename or ".npy" in filename: 
            pass
        elif "ID" not in filename:
            raise ValueError("Ops! Something went wrong...It seems that you must upload your data again! If this issue persists, please contact the app administrator.")
        else: 

            # Prepare data 
            print("Reading and evaluating your data...")
            
            # Extract the useful information from the file name to use them as dictionary keys 
            id = filename.split("_")[0][2:]
            s = filename.split("_")[1][1:]
            r = filename.split("_")[2][1:]
            download_date = filename.split("_")[4][:-4]

            # Create the dictionary for every patient, sensor and recording
            data_dict[id] = {}
            data_dict[id][s] = {}
            data_dict[id][s][r] = {}
            data_dict[id][s][r][download_date] = {}

            # Only read_csv is called if the file is .csv (only LibreView implemented so far)
            if filename.endswith(".csv") : 
                
                # Read the .csv and store it in a DataFrame. 
                current_recordings = pd.read_csv(filename, low_memory=False)

                # Clean NaN values
                current_recordings = current_recordings.dropna(axis=0, subset=['Tipo de registro'])

                # Clean errors detected: 
                # Detectederrors in the timestamps. These are removed
                # if id == "014" and s == "001" and r == "001" : 
                #     idxs = np.where(current_recordings['Sello de tiempo del dispositivo'] == '01-01-0001 00:00')
                #     current_recordings.drop(current_recordings.index[71870:74581], inplace=True)

                # Conver timestamps to datetime64
                current_recordings['Sello de tiempo del dispositivo'] = pd.to_datetime(current_recordings['Sello de tiempo del dispositivo'],
                                                                                    dayfirst=True,
                                                                                    format="%d-%m-%Y %H:%M",
                                                                                    exact=True)

                # Obtain sensors MACs (this is more robust that obtaining the sensor names, which has a lot of typos)
                MACs = current_recordings['Número de serial'].unique()

                # Iterate over the MACs, since it contains less errors than 'Dispositivo' column
                for i in range(0, len(MACs)) :

                    # Some instances could bring NaN serial number (MAC). These are discarded and not considered in further steps
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
        filename = 'libreview_data.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dict

def get_your_1year_LibreView_recordings_dict(your_keys : List, libreview_data : Dict) -> Dict: 

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

    # Initialize the dictionary entries
    data_1yr_recordings[your_keys[0]] = {}
    data_1yr_recordings[your_keys[0]][your_keys[1]] = {}
    data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]] = {}
    data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]] = {}

    for key in libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]].keys():
        
        # Initialize the entry regarding the MAC
        data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key] = {}

        for key2 in libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key].keys():

            # In case there are not CGM entries (could happen)
            if not libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["reading"].any():
                pass
            else: 

                # Calculate number of days so we can discard recordings with less than a year of data
                data_1st_sample = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["timestamp"][0]
                data_last_sample = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["timestamp"][-1]
                time_between_readings = data_last_sample - data_1st_sample

                if time_between_readings.days >= 365:

                    # Corresponding CGM readings and their timestamps
                    cgm = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["reading"]
                    cgm_timestamp = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["timestamp"]

                    
                    # Fill dictionary with readings of at least one year
                    data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2] = {"CGM" : {"reading" : cgm,
                                                                                                                            "timestamp" : cgm_timestamp}}

    # Iterate over all dictionary keys to delete the entries that are empty (meaning that they had <1 year of data )     
    for key in list(data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]].keys()):
    
            # Check if the entry is empty or not to delete it 
            if len(data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key]) == 0:

                # Delete entry 
                del data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key]

    # Check if the entry is empty (meaning that not enough data (i.e., 1 year) is nor present in the user), to delete the entry and to set a flag that 
    # will prompt the user the proper information 
    if len(data_1yr_recordings[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]]) == 0:

        # Delete entry 
        del data_1yr_recordings[your_keys[0]]

        data_suitability = False 
    else: 
        data_suitability = True 

    # Save dictionary as pickle
    with open('libreview_data_1yr_recordings.pickle', 'wb') as handle:
        pickle.dump(data_1yr_recordings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_1yr_recordings, data_suitability

def generate_your_LibreView_npy_files(your_data_path : str, your_keys : List,  libreview_data : Dict, verbose : int = 0) -> None: 
    
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
    
    filename = os.listdir(your_data_path)

    # Counter to be capable of index the same ID with every sensor
    id_MAC_combination = 0

    # The try except is to avoid the KeyError when there is no data for a given sensor
    try: 
        for key in libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]].keys():
            for key2 in libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key].keys():
                
                if verbose == 1 : 
                    print("Number of your CGM readings in sensor", key2,": ", libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["reading"].shape[0])
                
                summary_df.loc[id_MAC_combination, "Patient ID"] = your_keys[0]
                summary_df.loc[id_MAC_combination, "S"] = your_keys[1]
                summary_df.loc[id_MAC_combination, "R"] = your_keys[2]
                summary_df.loc[id_MAC_combination, "Sensor"] = key2

                if not libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["reading"].any():
                    
                    if verbose == 1: 
                        print("It seems you don't have readable CGM data with sensor", key2)
                    summary_df.loc[id_MAC_combination, "Data from 1st sample"] = "No CGM data"
                    summary_df.loc[id_MAC_combination, "Data from last sample"] = "No CGM data"
                    summary_df.loc[id_MAC_combination, "Nº of CGM samples"] = "No CGM data"
                
                else: 

                    summary_df.loc[id_MAC_combination, "Data from 1st sample"] = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["timestamp"][0]
                    summary_df.loc[id_MAC_combination, "Data from last sample"] = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["timestamp"][-1]
                    summary_df.loc[id_MAC_combination, "Days bewteen readings"] = summary_df.loc[id_MAC_combination, "Data from last sample"] - summary_df.loc[id_MAC_combination, "Data from 1st sample"]
                    summary_df.loc[id_MAC_combination, "Nº of CGM samples"] = libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["reading"].shape[0]
                
                # Update counter
                id_MAC_combination += 1
    except:
        pass
        
    # Counter to count the number of different sensors that a patient has used 
    num_of_ID_reading = 1

    # Same as before: try except to avoid the KeyError when there is no data for a given sensor
    try :

        num_of_ID_reading = 1

        for key in libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]].keys():
            for key2 in libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key].keys():
                filename = "X_" + your_keys[0] + "_" + str(num_of_ID_reading) + "_" + key2 + "_CGM.npy"
                timestamp_filename = "X_" + your_keys[0] + "_" + str(num_of_ID_reading) + "_" + key2 + "_CGM_timestamp.npy"
                
                # Save readings and timestamps as npy files
                np.save(filename, libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["reading"])
                np.save(timestamp_filename, libreview_data[your_keys[0]][your_keys[1]][your_keys[2]][your_keys[3]][key][key2]["CGM"]["timestamp"])

                # Increment counter
                num_of_ID_reading += 1
    except: 
        pass
            
def generate_your_LibreView_npy_1yr_recordings(data_1yr_recordings : Dict): 
    
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

                        # Set index to 1 because there are not more recordings for the current patient 
                        older_recording_idx = '1' 

                        # Read .npy files
                        recording = np.load('X_{}_{}_{}_CGM.npy'.format(id, older_recording_idx, sensor))
                        recording_timestamps = np.load('X_{}_{}_{}_CGM_timestamp.npy'.format(id, older_recording_idx, sensor), allow_pickle=True)

                        # Convert recording tp float 64
                        recording = recording.astype(np.float64)

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
                        np.save('oldest_1yr_CGM.npy', recording_1yr)
                        np.save('oldest_1yr_CGM_timestamp.npy', recording_timestamps_1yr)

def get_your_oldest_year_npys_from_LibreView_csv(dataset_path : str) -> bool: 

    """
    From the raw .csv files obtained from your LibreView, this function 
    generates a numpy files of your oldest year of CGM data of without interruptions.
    If you don't have at least one year of data, undortunately a model 
    will not be generated. For more information about how the data is extracted,
    please refer to the documentation of every particular function.

    If not enough data, prompts the user, teh program stops and provides information 
    about the IA so the user is able to upload better the next time 
    
    ######Files
    are stored in the '/1yr_npy_files' folder. CGM recordings are stored as
    'oldest_1yr_CGM.npy' and their timestamps as 'oldest_1yr_CGM_timestamp.npy' 
    ####################

    Args
    ----
        dataset_path : path where the .csv files are stored.
    
    Returns
    -------
        None 
    """
    # Go to the dataset directory 
    os.chdir(dataset_path)

    # Extract the ID from the uploaded file in the current user directory 
    filename = os.listdir(dataset_path)

    # Selection of the set of libreview keys after MANUALLY insert this entry in the code (TO BE IMPROVED AND STUDIED HOW TO CHANGE IT)
    your_id = filename[0].split("_")[0][2:] # filename[0] assumes only one element (the very first uploaded .csv)
    
    # Iterate over all dictionary keys
    for i in range(0,len(set_of_libreview_keys)):
        # Iterate until the ID matches the one of the list 
        if set_of_libreview_keys[i][0] == your_id: 
            your_keys = set_of_libreview_keys[i]
        else: 
            pass

    # This function is employed if you don't have your model. So this is the first time your data is read
    your_libreview_data = prepare_LibreView_data(dataset_path)

    # Take only the T1DM patients with at least one year in a row of CGM data with the same sensor 
    your_data_1yr_recordings, data_suitability = get_your_1year_LibreView_recordings_dict(your_keys, your_libreview_data)

    # Generate the Libreview .npy files from the generated or saved dictionary 
    generate_your_LibreView_npy_files(dataset_path, your_keys, your_data_1yr_recordings, r"/1yr_npy_files")

    # Extract an EXACT 1 year recordings from the dictionary and store them to load them separately 
    generate_your_LibreView_npy_1yr_recordings(your_data_1yr_recordings)

    return data_suitability 