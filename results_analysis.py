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

# results_analysis.py
# This module contains different functions to analyze the results of the experiments
# after their full execution. Graphs included in our paper has been generated using 
# these functions from a Jupyter Notebook. 
# See functions documentation for more details. 

from typing import Dict, List 
import os 
import json
import xlsxwriter
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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
                    Default: "C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files"
    
    Returns:
        None
    """

    # Go to the directory where all results are stored 
    os.chdir(results_dir)

    # Iterate over the ID folders to generate the 4-folds 
    for id in os.listdir(): 

        # Counter 
        i = 0

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id) and ('TEST' not in id) and ('h5' not in id) and ('xls' not in id) and ('evaluation' not in id) and ('pickle' not in id): 
        
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

                            if model == 'naive': 
                                pass

                            else:                            
                            
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
        exp_config (Dict): Dictionary with the experiment configuration (check training_configs.py) to be analyzed. 
        metrics (List): List with the metrics to be analyzed. Default: ['RMSE', 'MAE', 'MAPE', '% Parkes AB', '% ISO']
        results_dir (str): Directory where the folders with each patient ID and the results_dictionary.json are placed.
                    Default:
    
    Returns:
        grouped_metrics (Dict): Dictionary with the metrics grouped by model, PH and loss function. 
    """

    # Go to the directory where all results are stored 
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
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id) and ('TEST' not in id) and ('h5' not in id) and ('xls' not in id)and ('evaluation' not in id) and ('pickle' not in id): 
        
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

                            if model == 'naive': 
                                pass
                            
                            else: 

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
    Each studied subject has its own sheet in the corresponding directory. 
    Each PH (Prediction Horizon) has its own Excel file. Inside each file, there is one sheet per
    evaluated model. 

    Args:
    ----
        grouped_metrics (Dict): Dictionary containing all metrics organized, generated by group_best_patients_metrics()
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed (see training_configs.py).

    Returns:
    -------
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

def gen_PHs_boxplots(model_name : str, PH : int, iso_metrics : List, mse_metrics : List, legend : bool): 

    """
    Generate boxplots metric by metric separating the ISO-adapted loss from the MSE loss for a given model
    Args:
    ----
        model_name (str): model name 
        PH (int): Prediction Horizon
        iso_metrics : List with the metrics after training with the ISO-adapted loss function
        mse_metrics : List with the metrics after training with the MSE loss function
        legend (bool): If True, a legend is added to the plot.

    Returns:
    -------
        None
    """

    fig, ax1 = plt.subplots(figsize=(3, 3))

    metrics = ['RMSE', 'MAE', 'MAPE', 'Parkes', 'ISO']

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
    # text = 'Model: ' + model_name + ' - PH = ' + str(PH) + ' min'
    # fig.suptitle(text, fontsize=16, fontweight='bold')
    
    ax1.boxplot(mse_metrics, positions=np.arange(len(metrics))+space,
            boxprops=dict(facecolor='tab:gray'), **box_param)
    
    ax1.set_ylim(0, 55)

    # Set X ticks
    labelsize = 8
    ax1.set_xticks(np.arange(len(metrics)))
    ax1.set_xticklabels(metrics, fontsize=20)
    ax1.tick_params(axis='x', labelsize=labelsize)

    # Set size to the Y labels 
    ax1.tick_params(axis='y', labelsize=9)

    # Vertical line bewteen MAPE and Parkes
    ax1.axvline(x=2.5, c='black', linestyle='--')

    # Set legend 
    if legend:
        ax1.plot([], c='tab:green', label='ISO-adapted loss')
        ax1.plot([], c='tab:grey', label='MSE')
        # ax1.legend(fontsize=labelsize)
        ax1.legend(fontsize=7.5, loc='upper left')

    # Second axis to plot metrics with different range
    ax2 = ax1.twinx()

    # Set y limits in both axis
    
    ax2.boxplot(iso_metrics, positions=np.arange(len(metrics))-space,
                boxprops=dict(facecolor='tab:green'), **box_param)

    ax2.set_ylim(0, 105) # For ISO AND Parkes %
    
    # Set Y ticks and labels 
    # ax1.set_ylabel('MAPE, MAE, RMSE', fontsize=7)
    # ax2.set_ylabel('Parkes and ISO %', fontsize=7)
     
    # Set size to the Y labels 
    ax2.tick_params(axis='y', labelsize=9)
       
    # Save figure 
    plt.savefig('boxplot_'+model_name+'_PH_'+str(PH)+'_min.svg', format='svg', dpi=1200)

def get_patient_wise_metrics(exp_config : Dict, grouped_metrics : Dict, patients_data_folder : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files"): 
    
    """
    In order to gain insight about which patients are "better predicted"
    by the AI model, one figure per model containing one barplot per studied metric
    is generated. This studied the inter-subject variability within models. 

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
        if ('.npy' not in name) and ('.svg' not in name) and  ('.png' not in name) and ('TEST' not in name) and ('h5' not in name) and ('xls' not in name)and ('evaluation' not in name):
        
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

def get_grouped_RMSEbased_best_metrics(exp_config : Dict, grouped_metrics : Dict,
                                       patients_data_folder : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files"):

    """
    This functions analyses the best models for each patient and return a dictionary with 
    the best model, loss function and the best RMSE, MAE and MAPE metric value for each patient.
    These dictionaries  will be used for a clear results visualizarion.
    It also returns the IDs of the patients to access easily to their data. This function must be 
    improved. Sometimes it fails. But this function is not critical. 

    Args: 
    -----
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed.
        grouped_metrics (Dict): Dictionary containing all metrics organized, generated by group_best_patients_metrics()
        patients_data_folder: folder where the patients data are stored.
    
    Returns:
    --------
        best_30min_model_dict: dictionary with the best model, loss function and metric value for each patient for 30 mins.
        best_60min_model_dict: dictionary with the best model, loss function and metric value for each patient for 60 mins.
        patients_ids: list with the IDs of the patients.

    """

    # Go to the directory where all results are stored 
    # os.chdir(r"C:\Users\aralmeida\Downloads\LibreViewRawData\1yr_npy_files")
    os.chdir(patients_data_folder)

    # Empty dictionaries and list to store the best models for each patient
    best_30min_model_dict = {}
    best_60min_model_dict = {}
    patients_ids = []

    for id in os.listdir():

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id) and ('TEST' not in id) and ('h5' not in id) and ('xls' not in id)and ('evaluation' not in id) and ('pickle' not in id): 
                patients_ids.append(id)
                best_30min_model_dict[id] = {}
                best_60min_model_dict[id] = {}
    
    #####################################
    # Remove '008' '011' and '007' from list 
    # del best_30min_model_dict['008']
    # # del best_30min_model_dict['011']
    # del best_30min_model_dict['007']
    # del best_60min_model_dict['008']
    # del best_60min_model_dict['011']
    # del best_60min_model_dict['007']
    # patients_ids.remove('008')
    # patients_ids.remove('011')
    # patients_ids.remove('007')
    #####################################

    # For the metric that are better of they are lower (RMSE, MAE, MAPE)
    for i in range(len(patients_ids)): #num of available patients 

        # Fill the dictionaries once per patientd (ID)
        # best_30min_model_dict[patients_ids[i]] = {"samples" : 0, 
        #                                 "best_model_weights" : "",
        #                                 "RMSE": {"best_model" : "", "best_loss" : 0, "best_value" : 0}, 
        #                                 "MAPE": {"best_model" : "", "best_loss" : 0, "best_value" : 0},
        #                                 "MAE": {"best_model" : "", "best_loss" : 0, "best_value" : 0}}

        best_60min_model_dict[patients_ids[i]] = {"samples" : 0,
                                            "best_model_weights" : "",
                                            "RMSE": {"best_model" : "", "best_loss" : 0, "best_value" : 0}, 
                                            "MAPE": {"best_model" : "", "best_loss" : 0, "best_value" : 0},
                                            "MAE": {"best_model" : "", "best_loss" : 0, "best_value" : 0}} 

        ####### HARCODED #######
        # Go to the id directory 
        dir = 'C:\\Users\\aralmeida\\Downloads\\LibreViewRawData-final_sims\\1yr_npy_files\\' + patients_ids[i] + '\\N96\step1\\PH30\\multi\\month-wise-4-folds\\norm_min-max\\None_sampling\\DIL-1D-UNET\\ISO_loss'
        os.chdir(dir) 
        #######################

        # Load the X.npy that contains all the instances used to train the data 
        oldest_1yr_CGM = np.load('X.npy')
        shape = oldest_1yr_CGM.shape[0]

        # Iterate over the metrics to generate one graph per metric 
        for metric in ['RMSE', 'MAPE', 'MAE']:

            counter_30 = 0
            counter_60 = 0
            
            for models in exp_config["model"]:
                
                    for loss in ["ISO", "MSE"]: # harcoded: the studied loss functions 
                        
                        # Update the metric with each iteration
                        # 60 mins

                        # Naive is different because it doesn't have a loss function
                        if models == 'naive': 
                            curr_metric_60 = grouped_metrics[models]["60"][metric]["MSE"]["mean"][i]
                            curr_loss_func_60 = "MSE"
                            curr_model_60 = models

                            # curr_metric_30 = grouped_metrics[models]["30"][metric]["MSE"]["mean"][i]
                            # curr_loss_func_30 = "MSE"
                            # curr_model_30 = models
                        else: 
                            curr_metric_60 = grouped_metrics[models]["60"][metric][loss]["mean"][i]
                            curr_loss_func_60 = loss
                            curr_model_60 = models

                        if counter_30 == 0 and counter_60 == 0: 
                            
                            # best_metric_30 = curr_metric_30
                            # best_loss_func_30 = curr_loss_func_30
                            # best_model_30 = curr_model_30

                            best_metric_60 = curr_metric_60
                            best_loss_func_60 = curr_loss_func_60
                            best_model_60 = curr_model_60

                            # best_30min_model_dict[patients_ids[i]]["samples"] = shape 
                            # best_30min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_30
                            # best_30min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_30
                            # best_30min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_30

                            best_60min_model_dict[patients_ids[i]]["samples"] = shape
                            best_60min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_60
                            best_60min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_60
                            best_60min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_60

                            counter_30 = counter_30+1
                            # counter_60 = counter_60+1

                        else:
                            
                            # # For 30 mins
                            # if curr_metric_30 < best_metric_30:
                            #     best_metric_30 = curr_metric_30
                            #     best_loss_func_30 = curr_loss_func_30
                            #     best_model_30 = curr_model_30

                            #     best_30min_model_dict[patients_ids[i]]["samples"] = shape 
                            #     best_30min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_30
                            #     best_30min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_30                            
                            #     best_30min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_30

                            #     counter_30 = counter_30+1

                            # else:
                            #     best_metric_30 = best_metric_30 
                            #     best_loss_func_30 = best_loss_func_30
                            #     best_model_30 = best_model_30
            
                            #     best_30min_model_dict[patients_ids[i]]["samples"] = shape 
                            #     best_30min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_30
                            #     best_30min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_30
                            #     best_30min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_30

                            #     counter_30 = counter_30+1
                            
                            # For 60 mins
                            if curr_metric_60 < best_metric_60:
                                best_metric_60 = curr_metric_60
                                best_loss_func_60 = curr_loss_func_60
                                best_model_60 = curr_model_60


                                best_60min_model_dict[patients_ids[i]]["samples"] = shape 
                                best_60min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_60

                                counter_60 = counter_60+1
                            
                            else:
                                best_metric_60 = best_metric_60 
                                best_model_60 = best_model_60
                                best_loss_func_60 = best_loss_func_60
            
                                best_60min_model_dict[patients_ids[i]]["samples"] = shape 
                                best_60min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_60

                                counter_60 = counter_60+1
        
        os.chdir('..')
    
    return best_30min_model_dict, best_60min_model_dict, patients_ids

def get_grouped_ISO_and_Parkes_best_metrics(exp_config : Dict, grouped_metrics : Dict, 
                                            patients_data_folder : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files"):

    """
    This function analyses the best models for each patient and return a dictionary with 
    the best model, loss function and the best ISO and Parkes % metric value for each patient.
    These dictionaries  will be used for a clear results visualization.
    It also returns the IDs of the patients to access easily to their data. Some lines are 
    hardcoded with my directories. Please, check. 

    Args: 
    -----
        exp_config (Dict): Dictionary with the experiment configuration wanted to be analyzed.
        grouped_metrics (Dict): Dictionary containing all metrics organized, generated by group_best_patients_metrics()
        patients_data_folder: folder where the patients data are stored.
    
    Returns:
    --------
        best_30min_model_dict: dictionary with the best model, loss function and metric value for each patient for 30 mins.
        best_60min_model_dict: dictionary with the best model, loss function and metric value for each patient for 60 mins.
        patients_ids: list with the IDs of the patients.

    """

    # Go to the directory where all results are stored 
    os.chdir(patients_data_folder)

    # Empty dictionaries and list to store the best models for each patient
    best_30min_model_dict = {}
    best_60min_model_dict = {}
    patients_ids = []

    for id in os.listdir():

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id) and ('TEST' not in id) and ('h5' not in id) and ('xls' not in id) and ('evaluation' not in id) and ('pickle' not in id): 
                patients_ids.append(id)
                best_30min_model_dict[id] = {}
                best_60min_model_dict[id] = {}
    
    # For the metric that are better of they are lower (RMSE, MAE, MAPE)
    for i in range(len(patients_ids)): #num of available patients 

        # Fill the dictionaries once per patientd (ID)
        best_30min_model_dict[patients_ids[i]] = {"samples" : 0, 
                                        "best_model_weights" : "",
                                        "% Parkes AB": {"best_model" : "", "best_loss" : 0, "best_value" : 0}, 
                                        "% ISO": {"best_model" : "", "best_loss" : 0, "best_value" : 0}}

        best_60min_model_dict[patients_ids[i]] = {"samples" : 0,
                                            "best_model_weights" : "",
                                            "% Parkes AB": {"best_model" : "", "best_loss" : 0, "best_value" : 0}, 
                                            "% ISO": {"best_model" : "", "best_loss" : 0, "best_value" : 0}} 

        ####### HARCODED #######
        # Go to the id directory 
        dir = 'C:\\Users\\aralmeida\\Downloads\\LibreViewRawData-final_sims\\1yr_npy_files\\' + patients_ids[i] + '\\N96\step1\\PH30\\multi\\month-wise-4-folds\\norm_min-max\\None_sampling\\DIL-1D-UNET\\ISO_loss'
        os.chdir(dir) 
        #######################

        # Load the X.npy that contains all the instances used to train the data 
        oldest_1yr_CGM = np.load('X.npy')
        shape = oldest_1yr_CGM.shape[0]
    
        # Iterate over the metrics to generate one graph per metric 
        for metric in ['% Parkes AB', '% ISO']:

            counter_30 = 0
            counter_60 = 0
            
            for models in exp_config["model"]:
                
                    for loss in ["MSE", "ISO"]: # harcoded: the studied loss functions 
                        
                        # Update the metric with each iteration
                        # 30 mins
                        # curr_metric_30 = grouped_metrics[models]["30"][metric][loss]["mean"][i]
                        # curr_loss_func_30 = loss
                        # curr_model_30 = models
                        # 60 mins

                        if models == 'naive':
                            curr_metric_60 = grouped_metrics[models]["60"][metric]["MSE"]["mean"][i]
                            curr_loss_func_60 = "MSE"
                            curr_model_60 = models  

                            curr_metric_30 = grouped_metrics[models]["30"][metric]["MSE"]["mean"][i]
                            curr_loss_func_30 = "MSE"
                            curr_model_30 = models  
                        else:                          
                            curr_metric_60 = grouped_metrics[models]["60"][metric][loss]["mean"][i]
                            curr_loss_func_60 = loss
                            curr_model_60 = models

                            curr_metric_30 = grouped_metrics[models]["30"][metric][loss]["mean"][i]
                            curr_loss_func_30 = loss
                            curr_model_30 = models

                        if counter_30 == 0 and counter_60 == 0: 
                            
                            best_metric_30 = curr_metric_30
                            best_loss_func_30 = curr_loss_func_30
                            best_model_30 = curr_model_30

                            best_metric_60 = curr_metric_60
                            best_loss_func_60 = curr_loss_func_60
                            best_model_60 = curr_model_60

                            best_30min_model_dict[patients_ids[i]]["samples"] = shape 
                            best_30min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_30
                            best_30min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_30
                            best_30min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_30

                            best_60min_model_dict[patients_ids[i]]["samples"] = shape
                            best_60min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_60
                            best_60min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_60
                            best_60min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_60

                            counter_30 = counter_30+1
                            counter_60 = counter_60+1

                        else:
                            
                            # For 30 mins
                            if curr_metric_30 > best_metric_30:
                                best_metric_30 = curr_metric_30
                                best_loss_func_30 = curr_loss_func_30
                                best_model_30 = curr_model_30

                                best_30min_model_dict[patients_ids[i]]["samples"] = shape 
                                best_30min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_30
                                best_30min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_30                            
                                best_30min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_30

                                counter_30 = counter_30+1

                            else:
                                best_metric_30 = best_metric_30 
                                best_loss_func_30 = best_loss_func_30
                                best_model_30 = best_model_30
            
                                best_30min_model_dict[patients_ids[i]]["samples"] = shape 
                                best_30min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_30
                                best_30min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_30
                                best_30min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_30

                                counter_30 = counter_30+1
                            
                            # For 60 mins
                            if curr_metric_60 > best_metric_60:
                                best_metric_60 = curr_metric_60
                                best_loss_func_60 = curr_loss_func_60
                                best_model_60 = curr_model_60


                                best_60min_model_dict[patients_ids[i]]["samples"] = shape 
                                best_60min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_60

                                counter_60 = counter_60+1
                            
                            else:
                                best_metric_60 = best_metric_60 
                                best_model_60 = best_model_60
                                best_loss_func_60 = best_loss_func_60
            
                                best_60min_model_dict[patients_ids[i]]["samples"] = shape 
                                best_60min_model_dict[patients_ids[i]][metric]["best_model"] = best_model_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_loss"] = best_loss_func_60
                                best_60min_model_dict[patients_ids[i]][metric]["best_value"] = best_metric_60

                                counter_60 = counter_60+1
        
        os.chdir('..')
    
    return best_30min_model_dict, best_60min_model_dict, patients_ids

def get_global_MSEbased_graphics(PH : int, best_model_dict : Dict, ids : List) : 
    
    """
    Having the input of the PH to be studied and the dictionary with the best models
    plot the results joining metrics, models and patients. RMSE, MAPE and MAE results
    are plotted and stored.  

    Args:
    -----
        PH: prediction horizon to be studied. 
        best_model_dict: dictionary with the best models for each patient.
        ids: list with the patients' IDs.
    
    Returns:
    --------
        None
    """

    # Empty lists to be filled with the values of the best models for each patient 
    n_samples = []
    n_params = []
    loss_function_rmse= []
    loss_function_mape = []
    loss_function_mae = []
    rmse = []
    mape = []
    mae = []
    best_model_rmse = []
    best_model_mape = []
    best_model_mae = []

    for i in range(len(ids)): 
        
        n_samples.append(best_model_dict[ids[i]]["samples"])
        rmse.append(best_model_dict[ids[i]]["RMSE"]["best_value"])
        mape.append(best_model_dict[ids[i]]["MAPE"]["best_value"])
        mae.append(best_model_dict[ids[i]]["MAE"]["best_value"])
        loss_function_rmse.append(best_model_dict[ids[i]]["RMSE"]["best_loss"])
        loss_function_mape.append(best_model_dict[ids[i]]["MAPE"]["best_loss"])
        loss_function_mae.append(best_model_dict[ids[i]]["MAE"]["best_loss"])
        best_model_rmse.append(best_model_dict[ids[i]]["RMSE"]["best_model"])
        best_model_mape.append(best_model_dict[ids[i]]["MAPE"]["best_model"])
        best_model_mae.append(best_model_dict[ids[i]]["MAE"]["best_model"]) 

    # Plotting parameters 
    lf_shapes = {"ISO" : "o", "MSE" : "v"}
    model_color_dict = {'LSTM' : 'b', 
                        'StackedLSTM' : 'g',
                        'DIL-1D-UNET' : 'm',
                        'naive' : 'r'}

    # List of the list of the best metrics, loss functions and models 
    metrics = [rmse, mape, mae]
    loss_functions = [loss_function_rmse, loss_function_mape, loss_function_mae]
    models = [best_model_rmse, best_model_mape, best_model_mae]


    # Fill the list to properly plot the correspondant colours and shapes 
    model_rmse_colours = []
    model_mape_colours = []
    model_mae_colours = []

    for i in range(len(loss_function_rmse)):
        model_rmse_colours.append(model_color_dict[best_model_rmse[i]])
        model_mape_colours.append(model_color_dict[best_model_mape[i]])
        model_mae_colours.append(model_color_dict[best_model_mae[i]])

    models_colours = [model_rmse_colours, model_mape_colours, model_mae_colours]

    lf_rmse_shapes_30 = []
    lf_mape_shapes_30 = []
    lf_mae_shapes_30 = []

    for i in range(len(loss_function_rmse)):
        lf_rmse_shapes_30.append(lf_shapes[loss_function_rmse[i]])
        lf_mape_shapes_30.append(lf_shapes[loss_function_mape[i]])
        lf_mae_shapes_30.append(lf_shapes[loss_function_mae[i]])

    lf_shapes = [lf_rmse_shapes_30, lf_mape_shapes_30, lf_mae_shapes_30]

    ############### NUMBER OF PARAMS IS MISSING 
    num_params = 90
    ############### NUMBER OF PARAMS IS MISSING

    curr_metrics = ['RMSE', 'MAPE', 'MAE']

    for i in range(len(metrics)):

        plt.figure()

        for j in range(len(metrics[i])):

            # Plot the n_samples vs rmse, and the points are bigger if n_params is larger
            plt.scatter(n_samples[j], metrics[i][j], s = num_params, marker = lf_shapes[i][j], c=models_colours[i][j], label=metrics[i]) 
            plt.suptitle(curr_metrics[i] + ' - ' + str(PH) + ' mins', fontsize=16)
            plt.xlabel('Samples used to train and validate the model')

            if curr_metrics[i] == 'MAE': 
                plt.ylabel('MAE (mg/dL)', fontsize=14)
            elif curr_metrics[i] == 'RMSE':
                plt.ylabel('RMSE (mg/dL)', fontsize=14)
            elif curr_metrics[i] == 'MAPE':
                plt.ylabel('MAPE (%)', fontsize=14)

            # For legend plotting 
            custom = [plt.Line2D([], [], marker='.', color=model_color_dict['DIL-1D-UNET'], linestyle='None'),
                    plt.Line2D([], [], marker='.', color=model_color_dict['LSTM'], linestyle='None'), 
                    plt.Line2D([], [], marker='.', color=model_color_dict['StackedLSTM'], linestyle='None'),
                    plt.Line2D([], [], marker='.', color=model_color_dict['naive'], linestyle='None')]

            # Plot legend manually 
            plt.legend(handles = custom, labels=['DIL-1D-UNET', 'LSTM', 'StackedLSTM', 'naive'], loc='upper right', scatterpoints=1, frameon=False, labelspacing=1, title='Models', fontsize=10)

            # Save figure
            plt.savefig('global_results_' + str(PH) + 'mins_' + curr_metrics[i] + '.png', dpi=600, bbox_inches='tight')

def get_global_ISO_and_Parkes_graphics(PH : int, best_model_dict : Dict, ids : List) : 
    
    """
    Having the input of the PH to be studied and the dictionary with the best models
    plot the results joining metrics, models and patients. Parkes and ISO %
    metrics are plotted and stored.   

    Args:
    -----
        PH: prediction horizon to be studied. 
        best_model_dict: dictionary with the best models for each patient.
        ids: list with the patients' IDs.
    
    Returns:
    --------
        None
    """


    # Empty lists to be filled with the values of the best models for each patient 
    n_samples = []
    n_params = []
    loss_function_parkes = []
    loss_function_iso = []
    parkes = []
    iso = []
    best_model_parkes = []
    best_model_iso = []

    for i in range(len(ids)): 
        
        n_samples.append(best_model_dict[ids[i]]["samples"])
        parkes.append(best_model_dict[ids[i]]["% Parkes AB"]["best_value"])
        iso.append(best_model_dict[ids[i]]["% ISO"]["best_value"])
        loss_function_parkes.append(best_model_dict[ids[i]]["% Parkes AB"]["best_loss"])
        loss_function_iso.append(best_model_dict[ids[i]]["% ISO"]["best_loss"])
        best_model_parkes.append(best_model_dict[ids[i]]["% Parkes AB"]["best_model"])
        best_model_iso.append(best_model_dict[ids[i]]["% ISO"]["best_model"])


    # Plotting parameters 
    lf_shapes = {"ISO" : "o", "MSE" : "v"}
    model_color_dict = {'LSTM' : 'b', 
                        'StackedLSTM' : 'g',
                        'DIL-1D-UNET' : 'm',
                        'naive' : 'r'}

    # List of the list of the best metrics, loss functions and models 
    metrics = [parkes, iso]
    loss_functions = [loss_function_parkes, loss_function_iso]
    models = [best_model_parkes, best_model_iso]

    # Fill the list to properly plot the correspondant colours and shapes 
    model_parkes_colours = []
    model_iso_colours = []

    for i in range(len(loss_function_parkes)):
        model_parkes_colours.append(model_color_dict[best_model_parkes[i]])
        model_iso_colours.append(model_color_dict[best_model_iso[i]])

    models_colours = [model_parkes_colours, model_iso_colours]

    lf_parkes_shapes = []
    lf_iso_shapes = []


    for i in range(len(loss_function_parkes)):
        lf_parkes_shapes.append(lf_shapes[loss_function_parkes[i]])
        lf_iso_shapes.append(lf_shapes[loss_function_iso[i]])

    lf_shapes = [lf_parkes_shapes, lf_iso_shapes]

    ############### NUMBER OF PARAMS IS MISSING 
    num_params = 90
    ############### NUMBER OF PARAMS IS MISSING

    curr_metrics = ['% Parkes AB', '% ISO']

    for i in range(len(metrics)):

        plt.figure()

        for j in range(len(metrics[i])):

            # Plot the n_samples vs rmse, and the points are bigger if n_params is larger
            plt.scatter(n_samples[j], metrics[i][j], s = num_params, marker = lf_shapes[i][j], c=models_colours[i][j], label=metrics[i]) 
            plt.suptitle(curr_metrics[i] + ' - ' + str(PH) + ' mins', fontsize=16)
            plt.xlabel('Samples used to train and validate the model')

            if curr_metrics[i] == 'PARKES': 
                plt.ylabel('% Parkes AB', fontsize=14)
            elif curr_metrics[i] == 'ISO':
                plt.ylabel('% ISO', fontsize=14)
            
            custom = [plt.Line2D([], [], marker='.', color=model_color_dict['DIL-1D-UNET'], linestyle='None'),
                    plt.Line2D([], [], marker='.', color=model_color_dict['LSTM'], linestyle='None'), 
                    plt.Line2D([], [], marker='.', color=model_color_dict['StackedLSTM'], linestyle='None'),
                    plt.Line2D([], [], marker='.', color=model_color_dict['naive'], linestyle='None')]

            # Plot legend manually 
            plt.legend(handles = custom, labels=['DIL-1D-UNET', 'LSTM', 'StackedLSTM', 'naive'], loc='upper right', scatterpoints=1, frameon=False, labelspacing=1, title='Models', fontsize=10)

            # Save figure
            plt.savefig('global_results_' + str(PH) + 'mins_' + curr_metrics[i] + '.png', dpi=600, bbox_inches='tight')

def get_XY_vectors_characterization(patients_ids : List, PH : int, unit : str = 'mg/dL'):

    """
    In order to analyse the limitations of the DL models according to each patients, 
    some parameters are extracted from the CGM data and stored in a dictionary returned
    by this functions for X and Y vectors. This dictonary is ready to be manipulated to generate
    different figures. Prediction performance is studied against average CGM values, derivative and 
    second derivative values. Again, some of the code is hard-coded with some directories, 
    so please check it. 

    Args:
    -----
        patients_ids (List): List of patients IDs to be analysed.
        PH (int): Prediction horizon to be analysed.
        unit (str): Unit of the CGM data. Default: 'mg/dL'
    
    Returns: 
    --------
        cgm_patients_characteristics (Dict): Dictionary with the characteristics of the subjects' CGM data.

    """

    if unit == 'mg/dL':
        hyper_limit = 180
        hypo_limit = 70
        severe_hyper_limit = 250
        severe_hypo_limit= 54

    else: 
        hyper_limit = 10
        hypo_limit = 3.9
        severe_hyper_limit = 13.9
        severe_hypo_limit= 2.9

    cgm_patients_characteristics = {}

    for i in range(len(patients_ids)): #num of available patients 

        # Fill the dictionaries once per patientd (ID)
        cgm_patients_characteristics[patients_ids[i]] =  { "X" : {
                                                                    "max_values_avg" : 0,
                                                                    "min_values_avg" : 0,
                                                                    "derivative_max_values_avg" : 0,
                                                                    "derivative_min_values_avg" : 0,
                                                                    "num_windows_with_hyper" : 0,
                                                                    "num_windows_with_hypo" : 0, 
                                                                    "num_samples_with_hyper" : 0,
                                                                    "num_samples_with_hypo" : 0}, 
                                                                "Y" : {
                                                                    "max_values_avg" : 0,
                                                                    "min_values_avg" : 0,
                                                                    "derivative_max_values_avg" : 0,
                                                                    "derivative_min_values_avg" : 0,
                                                                    "num_windows_with_hyper" : 0,
                                                                    "num_windows_with_hypo" : 0,
                                                                    "num_samples_with_hyper" : 0,
                                                                    "num_samples_with_hypo" : 0}}

        ####### HARCODED #######
        # Go to the id directory 
        dir = 'C:\\Users\\aralmeida\\Downloads\\LibreViewRawData\\1yr_npy_files\\' + patients_ids[i] + '\\N144\step1\\PH' + str(PH) + '\\multi\\month-wise-4-folds\\norm_min-max\\None_sampling\\DIL-1D-UNET\\ISO_loss'
        os.chdir(dir) 

        # Files loading and analysis 
        # Load X and Y 
        X = np.load('X.npy')
        Y = np.load('Y.npy')

        # Get the maximum and minimum values of every instance 
        X_max_values = np.max(X, axis=1)
        X_min_values = np.min(X, axis=1)

        Y_max_values = np.max(Y, axis=1)
        Y_min_values = np.min(Y, axis=1)

        # Get the mean of the maximum and minimum values 
        X_mean_max = np.mean(X_max_values)
        X_mean_min = np.mean(X_min_values)

        Y_mean_max = np.mean(Y_max_values)
        Y_mean_min = np.mean(Y_min_values)

        # Obtain derivatives 
        X_deriv = np.diff(X, axis=1)
        Y_deriv = np.diff(Y, axis=1)

        # Get the maximum and minimum values of every instance of the derivative 
        X_deriv_max_values = np.max(X_deriv, axis=1)
        X_deriv_min_values = np.min(X_deriv, axis=1)

        Y_deriv_max_values = np.max(Y_deriv, axis=1)
        Y_deriv_min_values = np.min(Y_deriv, axis=1)

        # Get the mean of the maximum and minimum values of the derivative
        X_deriv_mean_max = np.mean(X_deriv_max_values)
        X_deriv_mean_min = np.mean(X_deriv_min_values)

        Y_deriv_mean_max = np.mean(Y_deriv_max_values)
        Y_deriv_mean_min = np.mean(Y_deriv_min_values)

        # Check instance by instance in X if there is a hyper value
        X_num_hyper_instances = 0
        X_num_hypo_instances = 0

        for instance in X:
            if np.any(instance > hyper_limit):
                X_num_hyper_instances += 1
            if np.any(instance < hypo_limit):
                X_num_hypo_instances  += 1

        # Count total samples with hyper and hypo
        X_hypo_samples = X[X < hypo_limit] # Para el numero total de muestras esta bien 
        X_num_hypo_samples = X_hypo_samples.shape[0]
        X_hyper_samples = X[X > hyper_limit]
        X_num_hyper_samples = X_hyper_samples.shape[0]

        # Check instance by instance in Y if there is a hyper value
        Y_num_hyper_instances = 0
        Y_num_hypo_instances = 0

        for instance in Y:
            if np.any(instance > hyper_limit):
                Y_num_hyper_instances += 1
            if np.any(instance < hypo_limit):
                Y_num_hypo_instances += 1

        # Count total samples with hyper and hypo
        Y_hypo_samples = Y[Y < hypo_limit] # Para el numero total de muestras esta bien 
        Y_num_hypo_samples = Y_hypo_samples.shape[0]
        Y_hyper_samples = Y[Y > hyper_limit]
        Y_num_hyper_samples = Y_hyper_samples.shape[0]

        # Fill the dictionary with the correspondant value 
        # X 
        cgm_patients_characteristics[patients_ids[i]]["X"]["max_values_avg"] = X_mean_max
        cgm_patients_characteristics[patients_ids[i]]["X"]["min_values_avg"] = X_mean_min
        cgm_patients_characteristics[patients_ids[i]]["X"]["derivative_max_values_avg"] = X_deriv_mean_max
        cgm_patients_characteristics[patients_ids[i]]["X"]["derivative_min_values_avg"] = X_deriv_mean_min
        cgm_patients_characteristics[patients_ids[i]]["X"]["num_windows_with_hyper"] = X_num_hyper_instances
        cgm_patients_characteristics[patients_ids[i]]["X"]["num_windows_with_hypo"] = X_num_hypo_instances
        cgm_patients_characteristics[patients_ids[i]]["X"]["num_samples_with_hyper"] = X_num_hyper_samples
        cgm_patients_characteristics[patients_ids[i]]["X"]["num_samples_with_hypo"] = X_num_hypo_samples
        # Y 
        cgm_patients_characteristics[patients_ids[i]]["Y"]["max_values_avg"] = Y_mean_max
        cgm_patients_characteristics[patients_ids[i]]["Y"]["min_values_avg"] = Y_mean_min
        cgm_patients_characteristics[patients_ids[i]]["Y"]["derivative_max_values_avg"] = Y_deriv_mean_max
        cgm_patients_characteristics[patients_ids[i]]["Y"]["derivative_min_values_avg"] = Y_deriv_mean_min
        cgm_patients_characteristics[patients_ids[i]]["Y"]["num_windows_with_hyper"] = Y_num_hyper_instances
        cgm_patients_characteristics[patients_ids[i]]["Y"]["num_windows_with_hypo"] = Y_num_hypo_instances
        cgm_patients_characteristics[patients_ids[i]]["Y"]["num_samples_with_hyper"] = Y_num_hyper_samples
        cgm_patients_characteristics[patients_ids[i]]["Y"]["num_samples_with_hypo"] = Y_num_hypo_samples

    return cgm_patients_characteristics

def get_patient_wise_fold_results(exp_config : Dict, results_dict : Dict, results_dir : str = r"C:\Users\aralmeida\Downloads\LibreViewRawData-final_sims\1yr_npy_files") -> Dict: 
    
    """
    This function returns a dictionary with the results of all folds for every patient. 
    This function assumes that all the results are stored in its correspondant dictionary. 
    Changes on the previously executed functions will likely affect this function. 

    Args:
    -----
        exp_config : Dictionary with the experiment configuration to be analyzed.
        results_dict : Dictionary with the results of the models for every patient (just to get the samples used to train the models)
        results_dir :  Directory where the results are stored.
       
    Returns:
    --------
        patient_wise_fold_results : Dictionary with the results of all folds for every patient. 
    """

    os.chdir(results_dir)

    patient_wise_fold_results = {}

    # Iterate over the ID folders to generate the 4-folds 
    for id in os.listdir(): 

        # Counter 
        i = 0

        # Consider only folders, not .npy or .txt files
        if ('npy' not in id) and ('txt' not in id) and ('svg' not in id) and ('png' not in id) and ('TEST' not in id) and ('h5' not in id) and ('xls' not in id) and ('evaluation' not in id) and ('pickle' not in id): 
        
            os.chdir(id)

            patient_wise_fold_results[id] = {}
            patient_wise_fold_results[id]['samples'] = results_dict[id]['samples']

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

                # Itreate over the loss functions and write the results
                for loss in exp_config['loss_function']:

                    patient_wise_fold_results[id][loss] = {}

                    for model in exp_config['model']:

                        if model == 'naive':
                            key = 'multi_N{}_step1_PH{}_month-wise-4-folds_min-max_None_{}_root_mean_squared_error'.format(exp_config['N'][0], PH, model, loss)
                        else: 
                            # Obtain key to access the correspondant result 
                            key = 'multi_N{}_step1_PH{}_month-wise-4-folds_min-max_None_{}_{}'.format(exp_config['N'][0], PH, model, loss)

                        patient_wise_fold_results[id][loss][model] = {'RMSE': [results[key][model]['1-fold']["normal "]["RMSE"][idx],
                                                                results[key][model]['2-fold']["normal "]["RMSE"][idx],
                                                                results[key][model]['3-fold']["normal "]["RMSE"][idx],
                                                                results[key][model]['4-fold']["normal "]["RMSE"][idx]], 
                                                                'ISO' : [results[key][model]['1-fold']["normal "]["ISO"][idx],
                                                                results[key][model]['2-fold']["normal "]["ISO"][idx],
                                                                results[key][model]['3-fold']["normal "]["ISO"][idx],
                                                                results[key][model]['4-fold']["normal "]["ISO"][idx]], 
                                                                'PARKES' : [results[key][model]['1-fold']["normal "]["PARKES"][idx],
                                                                results[key][model]['2-fold']["normal "]["PARKES"][idx],
                                                                results[key][model]['3-fold']["normal "]["PARKES"][idx],
                                                                results[key][model]['4-fold']["normal "]["PARKES"][idx]], 
                                                                'MAE' : [results[key][model]['1-fold']["normal "]["MAE"][idx],
                                                                results[key][model]['2-fold']["normal "]["MAE"][idx],
                                                                results[key][model]['3-fold']["normal "]["MAE"][idx],
                                                                results[key][model]['4-fold']["normal "]["MAE"][idx]],
                                                                'MAPE' : [results[key][model]['1-fold']["normal "]["MAPE"][idx],
                                                                results[key][model]['2-fold']["normal "]["MAPE"][idx],
                                                                results[key][model]['3-fold']["normal "]["MAPE"][idx],
                                                                results[key][model]['4-fold']["normal "]["MAPE"][idx]]#,
                                                                # 'Hypo TP' : [results[key][model]['1-fold']["normal "]["Hypo TP"],
                                                                # results[key][model]['2-fold']["normal "]["Hypo TP"],
                                                                # results[key][model]['3-fold']["normal "]["Hypo TP"],
                                                                # results[key][model]['4-fold']["normal "]["Hypo TP"]],
                                                                # 'Hyper TP' : [results[key][model]['1-fold']["normal "]["Hyper TP"],
                                                                # results[key][model]['2-fold']["normal "]["Hyper TP"],
                                                                # results[key][model]['3-fold']["normal "]["Hyper TP"],
                                                                # results[key][model]['4-fold']["normal "]["Hyper TP"]],
                                                                # 'Normal TP' : [results[key][model]['1-fold']["normal "]["Normal TP"],
                                                                # results[key][model]['2-fold']["normal "]["Normal TP"],
                                                                # results[key][model]['3-fold']["normal "]["Normal TP"],
                                                                # results[key][model]['4-fold']["normal "]["Normal TP"]], 
                                                                # 'Accuracy' : [results[key][model]['1-fold']["normal "]["Accuracy"],
                                                                # results[key][model]['2-fold']["normal "]["Accuracy"],
                                                                # results[key][model]['3-fold']["normal "]["Accuracy"],
                                                                # results[key][model]['4-fold']["normal "]["Accuracy"]]
                                                                }
                                                                    
            os.chdir('..')
    
    return patient_wise_fold_results

def plot_patient_per_patient_boxplot(patient_wise_results : Dict, metric : str, loss_function : str, PH : int, legend: bool, sorted_by : str = 'samples'):
    
    """ After the simulation of the DL model generation for 
    all patients (currently n=29) has been done, this function can 
    be executed to compare the different model performance depending on the patient,
    (i,.e., inter-subject and intra-subject variability).  
    This boxplot plot the metric (inputted as argument) for each patient. Patients 
    are sorted by their available samples to train the models. Boxplots represent mean and
    .std of the 4 folds for naive, LSTM, Stacked-LSTM and Dilated-1D-UNET models. The figures
    generated here were the ones included in our paper. Figure is saved considering PH, loss
    function and metric. 

    Args:
    -----
        patient_wise_results : Dictionary containing the results for each patient separated per folds and models.
        metric : Metric to be compared.
        loss_function: "ISO_loss" or "root_mean_squared_error"
        PH : Prediction horizon to be studied.
        legend : flag to plot the legend or not.
        sorted_by : Parameter to sort the patients. Default is 'samples' (number of samples available to train the models).
    """

    # Sort the dictionary depending on the input parameter 
    #     match sorted_by: 
    #           case 'samples':
    #                   patient_wise_results = dict(sorted(patient_wise_results.items(), key=lambda item: item[1]['samples'], reverse=True))
    
    patient_wise_results = dict(sorted(patient_wise_results.items(), key=lambda item: item[1]['samples'], reverse=True))

    DIL_60_rmse = []
    stacked_60_rmse = []
    naive_60_rmse = []
    LSTM_60_rmse = []
    samples = []

    for id in patient_wise_results.keys():
            DIL_60_rmse.append(patient_wise_results[id][loss_function]['DIL-1D-UNET'][metric])

            # Add in this step also samples
            samples.append(patient_wise_results[id]['samples'])

    # Convert rows to columns 
    DIL_60_rmse = np.array(DIL_60_rmse).T
    samples = np.array(samples).T

    # Create a dataframe
    DIL_60_rmse = pd.DataFrame(DIL_60_rmse, columns=patient_wise_results.keys()).assign(model='DIL-1D-UNET')

    for id in patient_wise_results.keys():
            stacked_60_rmse.append(patient_wise_results[id][loss_function]['StackedLSTM'][metric])

    # Convert rows to columns
    stacked_60_rmse = np.array(stacked_60_rmse).T

    # Create a dataframe
    stacked_60_rmse = pd.DataFrame(stacked_60_rmse, columns=patient_wise_results.keys()).assign(model='StackedLSTM')

    for id in patient_wise_results.keys():
            naive_60_rmse.append(patient_wise_results[id]['ISO_loss']['naive'][metric])

    # Convert rows to columns
    naive_60_rmse = np.array(naive_60_rmse).T

    # Create a dataframe
    naive_60_rmse = pd.DataFrame(naive_60_rmse, columns=patient_wise_results.keys()).assign(model='naive')

    for id in patient_wise_results.keys():
            LSTM_60_rmse.append(patient_wise_results[id][loss_function]['LSTM'][metric])

    # Convert rows to columns
    LSTM_60_rmse = np.array(LSTM_60_rmse).T

    # Create a dataframe
    LSTM_60_rmse = pd.DataFrame(LSTM_60_rmse, columns=patient_wise_results.keys()).assign(model='LSTM')

    # Declare figure and set size 
    fig = plt.figure()
    # fig.set_size_inches(20, 5) # LA BUENA
    fig.set_size_inches(18, 5) # LA BUENA

    # Concatenate and melt the dataframes to further plot them 
    concat_models = pd.concat([DIL_60_rmse, stacked_60_rmse, naive_60_rmse, LSTM_60_rmse])

    # Melt the dataframe
    melt_models = pd.melt(concat_models, id_vars=['model'], var_name=['Subject ID'])

    # Set color palette
    palette = {'DIL-1D-UNET': 'lightblue', 'StackedLSTM': 'lightsalmon', 'naive': 'silver', 'LSTM': 'yellow'}
    
    ax = sns.boxplot(x='Subject ID', y='value', data=melt_models, hue='model', palette=palette)

    # Draw the boxplot
    if metric == 'RMSE':
        y_label = 'RMSE (mg/dL)'
    elif metric == 'MAPE':
        y_label = 'MAPE (%)'
    elif metric == 'MAE':
        y_label = 'MAE (mg/dL)'
    elif metric == 'PARKES':
        y_label = 'ParkesAB (%)'
    elif metric == 'ISO':
        y_label = 'ISOZone (%)'

    # Set y label 
    ax.set_ylabel(y_label, fontsize = 16)

    # Second axis 
    ax2 = ax.twinx()

    # Add a line patient per patient in the background with the "samples" variable 
    ax2.plot(np.linspace(0, len(samples)-1, len(samples)), samples, 'o-', color='black', alpha=0.3)

    # Add y label for ax2
    ax2.set_ylabel('Nº of available samples', fontsize = 16)

    # Fill the background with a grey colour every patient 
    for i in range(len(samples)):
        if i % 2 == 0:
                ax.axvspan(i-0.5, i+0.5, facecolor='darkgrey', alpha=0.3)

    # Remove blank space on the left and right sides
    ax.set_xlim(-0.5, len(samples)-0.5)

    # When RMSE is evaluated, plot a dashed horizontal line in y = 32 with the text "state-of-the-art"
    if metric == 'RMSE':
        if PH == 60:
            # ax.axhline(y=32, color='black', linestyle='--')
            # ax.text(-0.2, 30.5, 'state-of-the-art', color = 'black')
            pass
        elif PH == 30:
            # ax.axhline(y=19, color='black', linestyle='--')
            # ax.text(-0.2, 18, 'state-of-the-art', color = 'black')
            pass
        else: 
            pass

    # When PARKES is evaluated, plot a dashed horizontal line in y = 99, since it is the minimum set by the ISO 
    elif metric == 'PARKES':
        ax.axhline(y=99, color='black', linestyle='--')
        # ax.text(-0.2, 100, 'ISO 15197:2015 compliance', color = 'black')

        # Change Y ticks depending on PH
        if PH == 30: 
            ax.set_yticks(np.arange(98, 100.5, 1))
        elif PH == 60: 
            ax.set_yticks(np.arange(86, 100.5, 2))

    # When ISO is evaluated, plot a dashed horizontal line in y = 95, since it is the minimum set by the ISO 
    elif metric == 'ISO':
        ax.axhline(y=95, color='black', linestyle='--')
        # ax.text(-0.2, 100, 'ISO 15197:2015 compliance', color = 'black')

    # Remove space from Y label to box 
    ax.yaxis.labelpad = 0

    # Add grid with horizontal black lines 
    ax.yaxis.grid(True, color='black')

    # Set X and Y ticks labels size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    # Set X label size 
    ax.set_xlabel('Subject ID', fontsize=16)

    if legend == True:

        ax.legend(loc='upper right')

        # Merge both legends
        handles, labels = ax.get_legend_handles_labels()

        # Declare a handle with the grey -o marker 
        grey_patch = mpatches.Patch(color='darkgrey', hatch='*', label='Nº of available samples')
        grey_patch.set_hatch('o')

        grey_point = Line2D([0], [0], label='Nº of available samples', marker='o', markersize=8, markeredgecolor='grey', markerfacecolor='grey', linestyle='-')

        # Set line to grey 
        grey_point.set_color('grey')

        ax.legend(handles + [grey_point], labels + ['Nº of available samples'], loc='upper right', fontsize=15)

    else: 

        # Delete all legends 
        ax.get_legend().remove()

    # Tight layout 
    plt.tight_layout()

    # Save figure 
    plt.savefig(str(PH)+'min_' + metric + loss_function + '.svg', format='svg', dpi=1200, bbox_inches='tight')