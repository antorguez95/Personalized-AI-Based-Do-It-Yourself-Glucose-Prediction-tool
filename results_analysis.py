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

from typing import Dict, List 
import os 
import json
import xlsxwriter
import numpy as np 
import matplotlib.pyplot as plt 

def store_ind_results_in_Excel(exp_config : Dict, results_dir : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files"):
    
    """
    Store the mean ± std. of the RMSE, MAE, MAPE, % ISO and % AB Parkes in an Excel file.
    The same metrics are shown for each one of the 4-folds. Notice that these are evaluation 
    metrics, not training metrics.
    This function assumes that there is a 'results_dictonary.json' file in each patient folder, and also 
    assumes that the directory is provided contains folders with the ID of each patients. If these assumptions
    are not correct, the function will not work.  
    Each studied patient has its own sheet on his/her directory. 
    Each PH (Prediction Horizon) has its own Excel file. Inside each file, there is one sheet per
    evaluated model. 

    Args:
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed. 
        results_dir (str): Directory where the folders with each patient ID and the results_dictionary.json are placed.
                    Default: 
    
    Returns:
        None
    """

    # Go to the directory where all results are stored 
    # os.chdir(r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files")
    os.chdir(results_dir)

    # Iterate over the ID folders to generate the 4-folds 
    for id in os.listdir(): 

        # Counter 
        i = 0

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id): 
        
            os.chdir(id)

            # Open json file
            with open('results_dictionary.json') as json_file:
                results = json.load(json_file)

            # Iterate over PH to generate different Excel files depending on the PH (not comparable between them)
            for PH in exp_config['PH']:

                # Filename
                filename = 'results_' + str(PH) + 'min.xlsx'

                # Index to access the dictionary results will vary because the greater PH, the greater the index
                if PH == 30:
                    idx = 1
                elif PH == 60:
                    idx = 3
                
                # Create an Excel file
                workbook = xlsxwriter.Workbook(filename)

                # Create one sheet per model 
                for model in  exp_config['model']: 

                    # Add worksheet corresponding to the current model 
                    worksheet = workbook.add_worksheet(model)

                    # Write the headers of the worksheet
                    worksheet.write(0, 0, 'Patient')
                    worksheet.write(0, 1, id)
                    
                    worksheet.write(1, 0, 'PH')
                    worksheet.write(1, 1,  PH)
                    
                    worksheet.write(4, 1, 'Loss function')
                    
                    worksheet.write(3, 2, 'RMSE')
                    worksheet.write(4, 2, 'MSE')
                    worksheet.write(4, 3, 'ISO')
                    worksheet.write(3, 4, 'MAE')
                    worksheet.write(4, 4, 'MSE')
                    worksheet.write(4, 5, 'ISO')
                    worksheet.write(3, 6, 'MAPE')
                    worksheet.write(4, 6, 'MSE')
                    worksheet.write(4, 7, 'ISO')
                    worksheet.write(3, 8, '% Parkes AB')
                    worksheet.write(4, 8, 'MSE')
                    worksheet.write(4, 9, 'ISO')
                    worksheet.write(3, 10, '% ISO')
                    worksheet.write(4, 10, 'MSE')
                    worksheet.write(4, 11, 'ISO')

                    worksheet.write(5, 1, '1-fold')
                    worksheet.write(6, 1, '2-fold')
                    worksheet.write(7, 1, '3-fold')
                    worksheet.write(8, 1, '4-fold')
                    worksheet.write(9, 1, 'mean ± std')

                    # Itreate over the loss functions and write the results
                    for loss in exp_config['loss_function']:

                        # Obtain key to access the correspondant result 
                        key = 'multi_N{}_step1_PH{}_month-wise-4-folds_min-max_None_{}_{}'.format(exp_config['N'][0], PH, model, loss)

                        if loss == 'root_mean_squared_error':

                            # RMSE
                            worksheet.write(5, 2, results[key][model]['1-fold']["normal "]["RMSE"][idx]) # typo: space after normal to be corrected
                            worksheet.write(6, 2, results[key][model]['2-fold']["normal "]["RMSE"][idx]) 
                            worksheet.write(7, 2, results[key][model]['3-fold']["normal "]["RMSE"][idx])
                            worksheet.write(8, 2, results[key][model]['4-fold']["normal "]["RMSE"][idx])
                            
                            # MAE
                            worksheet.write(5, 4, results[key][model]['1-fold']["normal "]["MAE"][idx]) # typo: space after normal to be corrected
                            worksheet.write(6, 4, results[key][model]['2-fold']["normal "]["MAE"][idx])
                            worksheet.write(7, 4, results[key][model]['3-fold']["normal "]["MAE"][idx])
                            worksheet.write(8, 4, results[key][model]['4-fold']["normal "]["MAE"][idx])

                            # MAPE
                            worksheet.write(5, 6, results[key][model]['1-fold']["normal "]["MAPE"][idx])
                            worksheet.write(6, 6, results[key][model]['2-fold']["normal "]["MAPE"][idx])
                            worksheet.write(7, 6, results[key][model]['3-fold']["normal "]["MAPE"][idx])
                            worksheet.write(8, 6, results[key][model]['4-fold']["normal "]["MAPE"][idx])

                            # % Parkes AB
                            worksheet.write(5, 8, results[key][model]['1-fold']["normal "]["PARKES"][idx])
                            worksheet.write(6, 8, results[key][model]['2-fold']["normal "]["PARKES"][idx])
                            worksheet.write(7, 8, results[key][model]['3-fold']["normal "]["PARKES"][idx])
                            worksheet.write(8, 8, results[key][model]['4-fold']["normal "]["PARKES"][idx])

                            # % ISO
                            worksheet.write(5, 10, results[key][model]['1-fold']["normal "]["ISO"][idx])
                            worksheet.write(6, 10, results[key][model]['2-fold']["normal "]["ISO"][idx])
                            worksheet.write(7, 10, results[key][model]['3-fold']["normal "]["ISO"][idx])
                            worksheet.write(8, 10, results[key][model]['4-fold']["normal "]["ISO"][idx])

                            # Mean and std
                            rmse_mean = np.mean([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['4-fold']["normal "]["RMSE"][idx]])
                            
                            rmse_std = np.std([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['4-fold']["normal "]["RMSE"][idx]])
                            
                            text = "{:.2f}".format(rmse_mean) + "±" + "{:.2f}".format(rmse_std)
                            worksheet.write(9, 2, text)
                            
                            mae_mean  = np.mean([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            mae_std  = np.std([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            text = "{:.2f}".format(mae_mean) + "±" + "{:.2f}".format(mae_std)
                            worksheet.write(9, 4, text)

                            mape_mean = np.mean([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            mape_std = np.std([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            text = "{:.2f}".format(mape_mean) + "±" + "{:.2f}".format(mape_std)
                            worksheet.write(9, 6, text)
                            
                            parkes_mean = np.mean([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])
                            
                            parkes_std = np.std([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])
                            
                            text = "{:.2f}".format(parkes_mean) + "±" + "{:.2f}".format(parkes_std)
                            worksheet.write(9, 8, text)

                            iso_mean = np.mean([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            iso_std = np.std([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            text = "{:.2f}".format(iso_mean) + "±" + "{:.2f}".format(iso_std)
                            worksheet.write(9, 10, text)
                            
                        elif loss == 'ISO_loss':                            
                            
                            # RMSE
                            worksheet.write(5, 3, results[key][model]['1-fold']["normal "]["RMSE"][idx]) # typo: space after normal to be corrected
                            worksheet.write(6, 3, results[key][model]['2-fold']["normal "]["RMSE"][idx])
                            worksheet.write(7, 3, results[key][model]['3-fold']["normal "]["RMSE"][idx])
                            worksheet.write(8, 3, results[key][model]['4-fold']["normal "]["RMSE"][idx])

                            # MAE
                            worksheet.write(5, 5, results[key][model]['1-fold']["normal "]["MAE"][idx])
                            worksheet.write(6, 5, results[key][model]['2-fold']["normal "]["MAE"][idx])
                            worksheet.write(7, 5, results[key][model]['3-fold']["normal "]["MAE"][idx])
                            worksheet.write(8, 5, results[key][model]['4-fold']["normal "]["MAE"][idx])

                            # MAPE
                            worksheet.write(5, 7, results[key][model]['1-fold']["normal "]["MAPE"][idx])
                            worksheet.write(6, 7, results[key][model]['2-fold']["normal "]["MAPE"][idx])
                            worksheet.write(7, 7, results[key][model]['3-fold']["normal "]["MAPE"][idx])
                            worksheet.write(8, 7, results[key][model]['4-fold']["normal "]["MAPE"][idx])

                            # % Parkes AB
                            worksheet.write(5, 9, results[key][model]['1-fold']["normal "]["PARKES"][idx])
                            worksheet.write(6, 9, results[key][model]['2-fold']["normal "]["PARKES"][idx])
                            worksheet.write(7, 9, results[key][model]['3-fold']["normal "]["PARKES"][idx])
                            worksheet.write(8, 9, results[key][model]['4-fold']["normal "]["PARKES"][idx])

                            # % ISO
                            worksheet.write(5, 11, results[key][model]['1-fold']["normal "]["ISO"][idx])
                            worksheet.write(6, 11, results[key][model]['2-fold']["normal "]["ISO"][idx])
                            worksheet.write(7, 11, results[key][model]['3-fold']["normal "]["ISO"][idx])
                            worksheet.write(8, 11, results[key][model]['4-fold']["normal "]["ISO"][idx])

                            # Mean and std
                            rmse_mean = np.mean([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['4-fold']["normal "]["RMSE"][idx]])
                            
                            rmse_std = np.std([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['4-fold']["normal "]["RMSE"][idx]])
                            
                            text = "{:.2f}".format(rmse_mean) + "±" + "{:.2f}".format(rmse_std)
                            worksheet.write(9, 3, text)

                            mae_mean  = np.mean([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            mae_std  = np.std([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            text = "{:.2f}".format(mae_mean) + "±" + "{:.2f}".format(mae_std)
                            worksheet.write(9, 5, text)

                            mape_mean = np.mean([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            mape_std = np.std([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            text = "{:.2f}".format(mape_mean) + "±" + "{:.2f}".format(mape_std)
                            worksheet.write(9, 7, text)

                            parkes_mean = np.mean([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])
                            
                            parkes_std = np.std([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])
                            
                            text = "{:.2f}".format(parkes_mean) + "±" + "{:.2f}".format(parkes_std)
                            worksheet.write(9, 9, text)

                            iso_mean = np.mean([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            iso_std = np.std([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            text = "{:.2f}".format(iso_mean) + "±" + "{:.2f}".format(iso_std)
                            worksheet.write(9, 11, text)

                # save excel file 
                workbook.close()

            os.chdir('..')

def group_best_patients_metrics(exp_config : Dict, metrics : List = ['RMSE', 'MAE', 'MAPE', '% Parkes AB', '% ISO'], results_dir : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files"):

    """

    This function groups all metrics for every model for each patient
    and stores their mean ± std. It return a dictionary with such 
    metrics. 

    Args:
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed. 
        metrics (List): List with the metrics to be analyzed. Default: ['RMSE', 'MAE', 'MAPE', '% Parkes AB', '% ISO']
        results_dir (str): Directory where the folders with each patient ID and the results_dictionary.json are placed.
                    Default:
    
    Returns:
        grouped_metrics (Dict): Dictionary with the metrics grouped by model, PH and loss function. 
    """

    # Go to the directory where all results are stored 
    # os.chdir(r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files")
    os.chdir(results_dir)

    # Dictionary to rearrange the metrics comfortably
    grouped_metrics = {}

    for model in exp_config['model']:
        grouped_metrics[model] = {"30" : {}, "60" : {}}
        for PH in exp_config['PH']:
            for metric in metrics: 
                grouped_metrics[model][str(PH)][metric] = {"ISO" : {"mean" : [], 
                                                                    "std" : []},
                                                            "MSE" : {"mean" : [],
                                                                    "std" : []}} 

    # Iterate over the ID folders to generate the 4-folds 
    for id in os.listdir(): 

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id): 
        
            os.chdir(id)

            # Open json file
            with open('results_dictionary.json') as json_file:
                results = json.load(json_file)

            # Iterate over PH to generate different Excel files depending on the PH (not comparable between them)
            for PH in exp_config['PH']:

                # Index to access the dictionary results will vary because the greater PH, the greater the index
                if PH == 30:
                    idx = 1
                elif PH == 60:
                    idx = 3
                
                # Create one sheet per model 
                for model in exp_config['model']: 

                    # Itreate over the loss functions and write the results
                    for loss in exp_config['loss_function']:

                        # Hard-coded to be changed 
                        N = exp_config['N'][0]
                        
                        # Obtain key to access the correspondant result 
                        key = 'multi_N{}_step1_PH{}_month-wise-4-folds_min-max_None_{}_{}'.format(N, PH, model, loss)                            

                        if loss == 'root_mean_squared_error':
                        
                            rmse_mean = np.mean([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                        results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                        results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                        results[key][model]['4-fold']["normal "]["RMSE"][idx]]) 

                            rmse_std = np.std([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                            results[key][model]['4-fold']["normal "]["RMSE"][idx]])                        

                            mae_mean  = np.mean([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            mae_std  = np.std([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            mape_mean = np.mean([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            mape_std = np.std([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            parkes_mean = np.mean([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])
                            
                            parkes_std = np.std([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])

                            iso_mean = np.mean([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            iso_std = np.std([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])                        

                            grouped_metrics[model][str(PH)]["RMSE"]["MSE"]["mean"].append(rmse_mean)
                            grouped_metrics[model][str(PH)]["MAPE"]["MSE"]["mean"].append(mape_mean)
                            grouped_metrics[model][str(PH)]["MAE"]["MSE"]["mean"].append(mae_mean)
                            grouped_metrics[model][str(PH)]["% Parkes AB"]["MSE"]["mean"].append(parkes_mean)
                            grouped_metrics[model][str(PH)]["% ISO"]["MSE"]["mean"].append(iso_mean) 

                            grouped_metrics[model][str(PH)]["RMSE"]["MSE"]["std"].append(rmse_std)
                            grouped_metrics[model][str(PH)]["MAPE"]["MSE"]["std"].append(mape_std)
                            grouped_metrics[model][str(PH)]["MAE"]["MSE"]["std"].append(mae_std)
                            grouped_metrics[model][str(PH)]["% Parkes AB"]["MSE"]["std"].append(parkes_std)
                            grouped_metrics[model][str(PH)]["% ISO"]["MSE"]["std"].append(iso_std)                                                           

                        elif loss == 'ISO_loss':

                            rmse_mean = np.mean([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                results[key][model]['4-fold']["normal "]["RMSE"][idx]])  

                            rmse_std = np.std([results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                results[key][model]['4-fold']["normal "]["RMSE"][idx]])                       

                            mae_mean  = np.mean([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            mae_std  = np.std([results[key][model]['1-fold']["normal "]["MAE"][idx],
                                            results[key][model]['2-fold']["normal "]["MAE"][idx],
                                            results[key][model]['3-fold']["normal "]["MAE"][idx],
                                            results[key][model]['4-fold']["normal "]["MAE"][idx]])
                            
                            mape_mean = np.mean([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            mape_std = np.std([results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                            results[key][model]['4-fold']["normal "]["MAPE"][idx]])
                            
                            parkes_mean = np.mean([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])
                            
                            parkes_std = np.std([results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                            results[key][model]['4-fold']["normal "]["PARKES"][idx]])

                            iso_mean = np.mean([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            iso_std = np.std([results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                            results[key][model]['4-fold']["normal "]["ISO"][idx]])
                            
                            
                            grouped_metrics[model][str(PH)]["RMSE"]["ISO"]["mean"].append(rmse_mean)
                            grouped_metrics[model][str(PH)]["MAPE"]["ISO"]["mean"].append(mape_mean)
                            grouped_metrics[model][str(PH)]["MAE"]["ISO"]["mean"].append(mae_mean)
                            grouped_metrics[model][str(PH)]["% Parkes AB"]["ISO"]["mean"].append(parkes_mean)
                            grouped_metrics[model][str(PH)]["% ISO"]["ISO"]["mean"].append(iso_mean)

                            grouped_metrics[model][str(PH)]["RMSE"]["ISO"]["std"].append(rmse_std)
                            grouped_metrics[model][str(PH)]["MAPE"]["ISO"]["std"].append(mape_std)
                            grouped_metrics[model][str(PH)]["MAE"]["ISO"]["std"].append(mae_std)
                            grouped_metrics[model][str(PH)]["% Parkes AB"]["ISO"]["std"].append(parkes_std)
                            grouped_metrics[model][str(PH)]["% ISO"]["ISO"]["std"].append(iso_std)  

            os.chdir('..')
    
    return grouped_metrics

def store_global_results_in_Excel(grouped_metrics : Dict, exp_config : Dict):
    
    """
    Store the mean ± std. of the RMSE, MAE, MAPE, % ISO and % AB Parkes
    for all patients in an Excel file.
    The same metrics are shown for each one of models. Notice that these are evaluation 
    metrics, not training metrics.
    This function assumes that there is a 'results_dictonary.json' file in each patient folder, and also 
    assumes that the directory is provided contains folders with the ID of each patients. If these assumptions
    are not correct, the function will not work.  
    Each studied model has its own sheet on his/her directory. 
    Each PH (Prediction Horizon) has its own Excel file. Inside each file, there is one sheet per
    evaluated model. 

    Args:
        grouped_metrics (Dict): Dictionary containing all metrics organized, generated by group_best_patients_metrics()
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed.


    Returns:
        None
    """

    # Create an unique Excel file with the overall results
    workbook = xlsxwriter.Workbook('global_results.xlsx')

    # Iterate over the PHs
    for PH in exp_config['PH']:

        # Model counters move in the Excel file
        row_counter = 5 # Begin with X due to the Excel file structure

        # Add worksheet corresponding to the PH
        sheetname = str(PH) + 'min'
        worksheet = workbook.add_worksheet(sheetname)

        # Loop to fill the file
        for model in exp_config['model']:

            # Write the headers of the worksheet
            worksheet.write(0, 0, 'PH')
            worksheet.write(0, 1,  str(PH)+'min')

            worksheet.write(4, 1, 'Loss function')

            worksheet.write(3, 2, 'RMSE')
            worksheet.write(4, 2, 'MSE')
            worksheet.write(4, 3, 'ISO')
            worksheet.write(3, 4, 'MAE')
            worksheet.write(4, 4, 'MSE')
            worksheet.write(4, 5, 'ISO')
            worksheet.write(3, 6, 'MAPE')
            worksheet.write(4, 6, 'MSE')
            worksheet.write(4, 7, 'ISO')
            worksheet.write(3, 8, '% Parkes AB')
            worksheet.write(4, 8, 'MSE')
            worksheet.write(4, 9, 'ISO')
            worksheet.write(3, 10, '% ISO')
            worksheet.write(4, 10, 'MSE')
            worksheet.write(4, 11, 'ISO')

            # Name of current model
            worksheet.write(row_counter, 1, model)

            if model != 'naive':

                # Compute mean and std of all metrics and place them
                # MSE loss
                mean = np.mean(grouped_metrics[model][str(PH)]["RMSE"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["RMSE"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 2, text)

                # ISO loss
                mean = np.mean(grouped_metrics[model][str(PH)]["RMSE"]["ISO"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["RMSE"]["ISO"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 3,text)

                # MSE loss
                mean = np.mean(grouped_metrics[model][str(PH)]["MAE"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["MAE"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 4, text)

                # ISO loss
                mean = np.mean(grouped_metrics[model][str(PH)]["MAE"]["ISO"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["MAE"]["ISO"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 5, text)

                # MSE loss
                mean = np.mean(grouped_metrics[model][str(PH)]["MAPE"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["MAPE"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 6, text)

                # ISO loss
                mean = np.mean(grouped_metrics[model][str(PH)]["MAPE"]["ISO"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["MAPE"]["ISO"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 7, text)

                # MSE loss
                mean = np.mean(grouped_metrics[model][str(PH)]["% Parkes AB"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["% Parkes AB"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 8, text)

                # ISO loss
                mean = np.mean(grouped_metrics[model][str(PH)]["% Parkes AB"]["ISO"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["% Parkes AB"]["ISO"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 9, text)

                # MSE loss
                mean = np.mean(grouped_metrics[model][str(PH)]["% ISO"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["% ISO"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 10, text)

                # ISO loss
                mean = np.mean(grouped_metrics[model][str(PH)]["% ISO"]["ISO"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["% ISO"]["ISO"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 11, text)
                
                # Once all results are written, increment counter
                row_counter = row_counter + 1
            
            elif model == 'naive': # We don't care about the loss function since this is not trainable

                # Compute mean and std of all metrics and place them
                mean = np.mean(grouped_metrics[model][str(PH)]["RMSE"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["RMSE"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 2, text)
                worksheet.write(row_counter, 3, '-')

                mean = np.mean(grouped_metrics[model][str(PH)]["MAE"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["MAE"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 4, text)
                worksheet.write(row_counter, 5, '-')

                mean = np.mean(grouped_metrics[model][str(PH)]["MAPE"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["MAPE"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 6, text)
                worksheet.write(row_counter, 7, '-')

                mean = np.mean(grouped_metrics[model][str(PH)]["% Parkes AB"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["% Parkes AB"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 8, text)

                mean = np.mean(grouped_metrics[model][str(PH)]["% ISO"]["MSE"]['mean'])
                std = np.std(grouped_metrics[model][str(PH)]["% ISO"]["MSE"]['mean'])
                text = "{:.2f}".format(mean) + "±" + "{:.2f}".format(std)
                worksheet.write(row_counter, 10, text)
                
                # Once all results are written, increment counter
                row_counter = row_counter + 1

    # Close current file
    workbook.close()

def gen_PHs_boxplots(model_name : str, PH : int, iso_metrics : List, mse_metrics : List): 

    fig, ax1 = plt.subplots(figsize=(9,6))

    metrics = ['RMSE', 'MAE', 'MAPE', '% Parkes AB', '% ISO']

    # Set font to arial 
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Set text to bold
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    box_param = dict(whis=(5, 95), widths=0.2, patch_artist=True,
                    flierprops=dict(marker='.', markeredgecolor='black',
                    fillstyle=None), medianprops=dict(color='black'))

    space = 0.15

    # Add title 
    text = 'Model: ' + model_name + ' - PH = ' + str(PH) + ' min'
    fig.suptitle(text, fontsize=16, fontweight='bold')
    
    ax1.boxplot(mse_metrics, positions=np.arange(len(metrics))+space,
            boxprops=dict(facecolor='tab:orange'), **box_param)
    

    # Second axis to plot metrics with different range
    ax2 = ax1.twinx()

    ax2.boxplot(iso_metrics, positions=np.arange(len(metrics))-space,
                boxprops=dict(facecolor='tab:blue'), **box_param)
    
    # Set y limits in both axis
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105) # For ISO AND Parkes %


    # Set X ticks
    labelsize = 12
    ax1.set_xticks(np.arange(len(metrics)))
    ax1.set_xticklabels(metrics)
    ax1.tick_params(axis='x', labelsize=labelsize)

    # Set Y ticks and labels 
    ax1.set_ylabel('MAPE, MAE, RMSE')
    ax2.set_ylabel('Parkes and ISO %')
    
    # ax2.set_yticks(ax2.get_yticks())
    
    # Set legend 
    ax1.plot([], c='tab:blue', label='ISO')
    ax1.plot([], c='tab:orange', label='MSE')
    ax1.legend(fontsize=labelsize)

    # Vertical line bewteen MAPE and Parkes
    ax1.axvline(x=2.5, c='black', linestyle='--')


def get_patient_wise_metrics(exp_config : Dict, grouped_metrics : Dict, patients_data_folder : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files"): 
    
    """
    In order to gain insight about which patients are less or more "predictable"
    by the AI model, one figure per model containing one barplot per studied metric
    is generated.

    Args: 
    ----
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed.
        grouped_metrics (Dict): Dictionary containing all metrics organized, generated by group_best_patients_metrics()
        patients_data_folder : Path to the folder containing the patients' data folders
    
    Returns:
    -------
        None
    """

    # os.chdir(r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files")
    os.chdir(patients_data_folder)

    # Get the files from dir 
    files = os.listdir() 
    patients_ids = []

    for name in files: 

        # Only add names that are folders 
        if ('.npy' not in name) and ('.svg' not in name) and  ('.png' not in name):
        
            # Add the name 
            patients_ids.append(name)
        
    # Get the number of patients 
    n_patients = len(patients_ids)

    # Vector of 29 positions to plot
    patients_vector = np.arange(1, n_patients + 1)

    # Get one figure per model-PH combination 
    for model in exp_config['model']: 
        for PH in exp_config['PH']: 

            fig, ax = plt.subplots(3, 2, figsize=(15,15))

            # Plot bar diagram for all patients of the metric 
            plt.figure()

            plt.suptitle(model + ' - PH = ' + str(PH) + ' min: ')
            ax[0,0].bar(patients_vector, grouped_metrics[model][str(PH)]["RMSE"]["ISO"]["mean"], yerr = grouped_metrics[model][str(PH)]["RMSE"]["ISO"]["std"], color='tab:blue')
            ax[0,0].set_xlabel("RMSE")

            ax[0,1].bar(np.arange(1, 30), grouped_metrics[model][str(PH)]["MAPE"]["ISO"]["mean"], yerr = grouped_metrics[model][str(PH)]["MAPE"]["ISO"]["std"], color='tab:blue')
            ax[0,1].set_xlabel("MAPE")

            # Remove ax[1,1]
            fig.delaxes(ax[1,1])

            # Center and set size 
            ax[1,0].set_position([0.300, 0.365, 0.350, 0.25])

            ax[1,0].bar(np.arange(1, 30), grouped_metrics[model][str(PH)]["MAE"]["ISO"]["mean"], yerr = grouped_metrics[model][str(PH)]["MAE"]["ISO"]["std"], color='tab:blue')
            ax[1,0].set_xlabel("MAE")

            ax[2,0].bar(np.arange(1, 30), grouped_metrics[model][str(PH)]["% Parkes AB"]["ISO"]["mean"], yerr = grouped_metrics[model][str(PH)]["% Parkes AB"]["ISO"]["std"], color='tab:blue')
            ax[2,0].set_xlabel("% Parkes AB")

            ax[2,1].bar(np.arange(1, 30), grouped_metrics[model][str(PH)]["% ISO"]["ISO"]["mean"], yerr = grouped_metrics[model][str(PH)]["% ISO"]["ISO"]["std"], color='tab:blue')
            ax[2,1].set_xlabel("% ISO")

            # Title
            fig.suptitle(model + ' - PH = ' + str(PH) + ' min: ')

            # Save figure
            plt.savefig('patient-wise_'+model + '_PH_' + str(PH) + '_min.png', dpi=1200) 