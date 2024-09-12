# Copyright (C) 2023 Antonio Rodriguez and Himar Fabelo
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

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import tensorflow as tf
from typing import Dict
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import pandas as pd


from models.training import ISO_adapted_loss
from models.multi_step.naive_model import naive_model 

from evaluation.multi_step.Parkes_EGA_boundaries_T1DM import *

from utils import generate_ranges_tags


def bgISOAcceptableZone(ground_truth : np.array, predictions: np.array,  fold : str, step : int, plot : bool = False) -> Tuple[int, bool]:
    
    """
    This function generates a chart showing the ISO range acceptable zone to evaluate 
    blood glucose prediction algorithms according to the ISO 15197:2015 (In vitro 
    diagnostic test systems - Requirements for blood-glucose monitoring systems for 
    self-testing in managing diabetes mellitus) [1]. It shows all predictions within
    and shows also the percentage of predictions within the acceptable range that
    must be >= 95%. Since this function can be used in the callbacks of the Deep Learning
    model, plot flag should be set to False to avoid memory issues with the generated
    figures. 

    Args:
    -----
        ground_truth: array with the ground truth to be compared with the predictions
        predictions: array with the predictions of glucose values of a given model
        fold : if many folds evaluated separately, indicate it so figures are properly saved
        plot: boolean indicating if the plot must be shown or not
        
    Returns:
    --------
        percentage: percentage of predicted points in the acceptable zone
        acceptability: boolean indicating if the percentage is acceptable or not 
    
    References:
    -----------
    [1] ISO 15197:2015
    
    """

    # Compute the error between the predictions and the ground truth
    # ISO: 95% of values must be +-15 mg/dL when < 100 mg/dL and +-15% when >=100 mg/dL
    error = ground_truth - predictions

    # Upper limits
    region_x = [0,100,500]
    regionUp_y = [15,15,75]

    # Bottom limits
    regionDown_y = [-15,-15,-75]

    # Number of samples to compute the percentage
    total_samples = len(error)

    # Store the CGM values < 100 mg/dL in a separated variable 
    cgm_lower_range = ground_truth < 100

    # Same with the higher range 
    cgm_higher_range = ground_truth >= 100

    # Compute the absolute error for both ranges 
    abs_error_lower_range = np.abs(error[cgm_lower_range])
    abs_error_higher_range = np.abs(error[cgm_higher_range])

    # Identify the values out of the lower range
    out_of_limits_lower_range = abs_error_lower_range > 15

    # Copmute the total percentage of samples out of range in the first range
    first_percent_out = np.sum(out_of_limits_lower_range) / total_samples * 100

    # Extract the ground truth values correspondant to the higher range 
    grount_truth_higher_range = ground_truth[cgm_higher_range]

    # Compute the percentage limit 15%
    percentage_limit = 0.15 * grount_truth_higher_range

    # Identify values higher than 15% in the high range 
    out_of_limits_higher_range = abs_error_higher_range > percentage_limit

    # Compute the total percentage of samples out of range in the second range 
    second_percent_out = np.sum(out_of_limits_higher_range) / total_samples * 100

    # Sum both percentages
    percentages_out = first_percent_out + second_percent_out

    # Compute the percentages in 
    percentage_in = 100 - percentages_out

    if percentage_in >= 95:
        acceptability = True
    else: 
        acceptability = False

    # Only plot if flag set to True 
    if plot == True:

        # Figure
        plt.figure(figsize=(5,5))

        # Set X and Y limits
        plt.xlim([0,550])
        plt.ylim([-90,90])

        # Set X and Y labels
        plt.xlabel('Glucose reference (mg/dL)', fontsize=14)
        plt.ylabel('Prediction error (mg/dL)', fontsize=14)

        # Set X and Y label size
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Plot boundaries in the figure
        plt.plot(region_x,regionUp_y, '--r')
        plt.plot(region_x,regionDown_y, '--r')

        # Plot error points and label 
        plt.plot(ground_truth, error,'b.')
    
        # Insert the percentage in within the acceptable range white background in the text box
        plt.text(0.05, 0.95, 'Percentage in acceptable zone: ' + str(round(percentage_in,2)) + '%', transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        # Set title 
        plt.title('ISO 15197:2015 for ' + str(step+1) + ' time step')
        
        # Save the figure
        plt.savefig(str(fold)+'_ISO15197.svg', dpi=600, bbox_inches='tight')
        # plt.savefig('fig3a.svg', dpi=600, bbox_inches='tight')
    
    return percentage_in, percentages_out, acceptability
 
def parkes_EGA_identify(ground_truth : np.array, predictions : np.array, str : int, unit : str = "mg_dl") -> Tuple[int, int, int, int, int, int] : 
    """
    Code adapted from Matlab code by Rupert Thomas, 2016. Quoting the original author:

    "For performance assessment of a method of blood glucose concentration
    measurement (input_y), against a reference method (input_x)
    
    N.B.: this tool is provided 'as-is', with no warranty. Don't use this for
    anything important/clinical without thoroughly double-checking the code
    and making sure it does what you expect it to do!
    
    Determines which area of the Parkes error grid the supplied sample data
    resides. input_x is a measurement (or vector of measurements) from the 
    reference method (comparator, e.g. gold standard), and input_y is a measurement
    or vector) from the instrument of interest".
    
    By default, the units of glucose concentration are in mg/dl. This is also
    the case if a value is not supplied for the input parameter units_mg_dl.
    If units_mg_dl is set to false, units of mili-molar (mM) are used
    instead.
    
    In the original Parkes paper, the grid coordinates were not
    mathematically specified, so the data here has been taken from:
    'Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013"
    
    Args:
    -----
        ground_truth: array with the ground truth to be compared with the predictions
        predictions: array with the predictions of glucose values of a given model
        units : string with the units of the glucose concentration. Default is mg/dl. Can be switch 
        to mM by setting units = "mM"

    Returns:
    --------
        inA: vector identiying if a particular point is in region A of the Parkes error grid
        inB: vector identiying if a particular point is in region B of the Parkes error grid
        inC: vector identiying if a particular point is in region C of the Parkes error grid
        inD: vector identiying if a particular point is in region D of the Parkes error grid
        inE: vector identiying if a particular point is in region E of the Parkes error grid
        OOR: vector identiying if a particular point is out of range of the Parkes error grid

    
    References:
    -----------
    [1] 'The Parkes Error Grid' - Parkes, 1994'

    [2] Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013 
    
    """

    ##### SUPPOSED TO BE MOVED OUTSIDE ########
    # Boundary vertices for Parkes Error Grid Analysis
    #  (Type 1 diabetes and regulatory use)

    # Rupert Thomas, 2016

    # N.B.: this data is provided 'as-is', with no warranty. Don't use this for
    # anything important/clinical without thoroughly double-checking the code
    # and making sure it does what you expect it to do!

    # Source:
    # 'Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013 

    # Region A
    regionA_x = [0,50,50,170,385,550,550,430,280,140,30,0]
    regionA_y = [0,0, 30,145,300,450,550,550,380,170,50,50]

    # Region B
    regionB_x = [0,120, 120,260,550,550,260, 70,50,30, 0]
    regionB_y = [0,  0,  30,130,250,550,550,110,80,60,60]

    # Region C
    regionC_x = [0,250, 250,550,550,125, 80, 50, 25,  0]
    regionC_y = [0,  0,  40,150,550,550,215,125,100,100]

    # Region D
    regionD_x = [0,550,550,50,  35,  0]
    regionD_y = [0,  0,550,550,155,150]

    # Region E - everything else in range
    regionE_x = [0,  0,550,550]
    regionE_y = [0,550,550,  0]
    ###########################################

    # Set flag to true if units are in mg/dl
    if unit == "mg_dl":
        units_mg_dl = True
    elif unit == "mM":
        units_mg_dl = False
    else: 
        units_mg_dl = True

    # If units are mM, convert boundaries
    if units_mg_dl == False:
        regionA_x = regionA_x * 0.05556
        regionA_y = regionA_y * 0.05556
        regionB_x = regionB_x * 0.05556
        regionB_y = regionB_y * 0.05556
        regionC_x = regionC_x * 0.05556
        regionC_y = regionC_y * 0.05556
        regionD_x = regionD_x * 0.05556
        regionD_y = regionD_y * 0.05556
        regionE_x = regionE_x * 0.05556
        regionE_y = regionE_y * 0.05556

    # Find points in each polygon region (adapted from in_polygo Matlab function)
    points = np.array([ground_truth, predictions]).T
    polygonA = Polygon(np.array([regionA_x, regionA_y]).T)
    inA = [polygonA.contains(Point(p)) for p in points]

    polygonB = Polygon(np.array([regionB_x, regionB_y]).T)
    inB = [polygonB.contains(Point(p)) for p in points]

    polygonC = Polygon(np.array([regionC_x, regionC_y]).T)
    inC = [polygonC.contains(Point(p)) for p in points]

    polygonD = Polygon(np.array([regionD_x, regionD_y]).T)
    inD = [polygonD.contains(Point(p)) for p in points]

    polygonE = Polygon(np.array([regionE_x, regionE_y]).T)
    inE = [polygonE.contains(Point(p)) for p in points]

    # Convert to numpy arrays
    inA = np.array(inA, dtype=bool)
    inB = np.array(inB, dtype=bool)
    inC = np.array(inC, dtype=bool)
    inD = np.array(inD, dtype=bool)
    inE = np.array(inE, dtype=bool)

    # Remove ovelaps: this must be done backwards -E = E and (not D)
    inE = np.logical_and(inE, np.logical_not(inD))
    inD = np.logical_and(inD, np.logical_not(inC))
    inC = np.logical_and(inC, np.logical_not(inB))
    inB = np.logical_and(inB, np.logical_not(inA))

    # Generate vector of out of range values
    OOR = np.logical_not(np.logical_or(np.logical_or(np.logical_or(inA,inB), inC), inD), inE)
    
    return inA, inB, inC, inD, inE, OOR

def parkes_EGA_chart(ground_truth : np.array, predictions : np.array, fold : str, step : int, unit : str = "mg_dl"):
    """
    Function to plot the Parkes Error Grid Analysis chart. It depends on 
    `parkes_EGA_identify` function that computes all the results. 

    
    Args:
    -----


    Returns:
    --------

    
    References:
    -----------
    [1] 'The Parkes Error Grid' - Parkes, 1994'

    [2] Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013 
    
    """

     ##### SUPPOSED TO BE MOVED OUTSIDE ########
    # Boundary vertices for Parkes Error Grid Analysis
    #  (Type 1 diabetes and regulatory use)

    # Rupert Thomas, 2016

    # N.B.: this data is provided 'as-is', with no warranty. Don't use this for
    # anything important/clinical without thoroughly double-checking the code
    # and making sure it does what you expect it to do!

    # Source:
    # 'Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013 

    # Region A
    regionA_x = [0,50,50,170,385,550,550,430,280,140,30,0]
    regionA_y = [0,0, 30,145,300,450,550,550,380,170,50,50]

    # Region B
    regionB_x = [0,120, 120,260,550,550,260, 70,50,30, 0]
    regionB_y = [0,  0,  30,130,250,550,550,110,80,60,60]

    # Region C
    regionC_x = [0,250, 250,550,550,125, 80, 50, 25,  0]
    regionC_y = [0,  0,  40,150,550,550,215,125,100,100]

    # Region D
    regionD_x = [0,550,550,50,  35,  0]
    regionD_y = [0,  0,550,550,155,150]

    # Region E - everything else in range
    regionE_x = [0,  0,550,550]
    regionE_y = [0,550,550,  0]
    ###########################################   

    # Set flag to true if units are in mg/dl
    if unit == "mg_dl":
        units_mg_dl = True
    elif unit == "mM":
        units_mg_dl = False
    
    # Set flag to true if units are in mg/dl
    if unit == "mg_dl":
        units_mg_dl = True
    elif unit == "mM":
        units_mg_dl = False
    
    # If units are mM, convert boundaries
    if units_mg_dl == False:
        regionA_x = regionA_x * 0.05556
        regionA_y = regionA_y * 0.05556
        regionB_x = regionB_x * 0.05556
        regionB_y = regionB_y * 0.05556
        regionC_x = regionC_x * 0.05556
        regionC_y = regionC_y * 0.05556
        regionD_x = regionD_x * 0.05556
        regionD_y = regionD_y * 0.05556
        regionE_x = regionE_x * 0.05556
        regionE_y = regionE_y * 0.05556
    
    # Indentification of the points in the Parkes Error Grid
    inA, inB, inC, inD, inE, OOR = parkes_EGA_identify(ground_truth, predictions, units_mg_dl)

    # Compute how many points are in each region
    points_in_regions = np.array([sum(inA), sum(inB), sum(inC), sum(inD), sum(inE), sum(OOR)])

    # Percentage of values in each region
    percentage_values = points_in_regions/len(ground_truth)*100
    percentage_AB = percentage_values[0] + percentage_values[1]

    # Get boundary data 
    # Region A
    regionA_x = [0,50,50,170,385,550,550,430,280,140,30,0]
    regionA_y = [0,0, 30,145,300,450,550,550,380,170,50,50]

    # Region B
    regionB_x = [0,120, 120,260,550,550,260, 70,50,30, 0]
    regionB_y = [0,  0,  30,130,250,550,550,110,80,60,60]

    # Region C
    regionC_x = [0,250, 250,550,550,125, 80, 50, 25,  0]
    regionC_y = [0,  0,  40,150,550,550,215,125,100,100]

    # Region D
    regionD_x = [0,550,550,50,  35,  0]
    regionD_y = [0,  0,550,550,155,150]

    # Region E - everything else in range
    regionE_x = [0,  0,550,550]
    regionE_y = [0,550,550,  0]

    units_mg_dl = True
    
    # Plot Parker Chart  
    plt.figure(figsize=(5,5))

    # Set X and Y limits
    plt.xlim([0,550])
    plt.ylim([0,550]) 

    # Set X an Y labels
    plt.xlabel('Glucose reference (mg/dL)', fontsize=14)
    plt.ylabel('Prediction error (mg/dL)', fontsize=14)

    # Set X and Y label size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Plot boundaries
    plt.plot(regionA_x,regionA_y, '-k')
    plt.plot(regionB_x,regionB_y, '-r')
    plt.plot(regionC_x,regionC_y, '-b')
    plt.plot(regionD_x,regionD_y, '-g')
    plt.plot(regionE_x,regionE_y, '-c')

    # Plot points and labels
    plt.plot(ground_truth, predictions, 'b.')

    # Letter X and Y 
    lettersX = [450, 250, 450, 150, 450, 450, 75, 20 ]
    lettersY = [450, 450, 250, 450, 150, 75, 450, 450]
    letters = ['A','B','B','C','C','D','D','E']

    # Draw letters 
    for i in range(0, len(lettersX)):
        plt.text(lettersX[i], lettersY[i], letters[i], fontsize=14, color='k')

    # Set background color as transparent green in AB zone 
    plt.fill_between(regionB_x, regionB_y, color='g', alpha=0.3)

    # Set auxiliary points to fill only EB zone
    aux_X = [0,30,50,70,260]
    aux_Y1 = [60,60,80,110,550]
    aux_Y2 = [550,550,550,550,550]
    plt.fill_between(aux_X, aux_Y1, aux_Y2, color='r', alpha=0.3)
 
    # Same for BD zone 
    aux_X = [120,260,550]
    aux_Y1 = [0,0,0]
    aux_Y2 = [30,130,250]
    plt.fill_between(aux_X, aux_Y1, aux_Y2, color='r', alpha=0.3)
    
    # Insert the percentage in within the acceptable range white background in the text box
    plt.text(0.05, 0.95, 'Percentage in AB: ' + str(round(percentage_AB,2)) + '%', transform=plt.gca().transAxes, fontsize=14,
    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Set title
    plt.title('Parkes Error Grid Analysis for ' + str(step+1) + ' time step')

    # Save the figure
    # plt.savefig(str(fold)+'_ParkerGrid.svg', dpi=600, bbox_inches='tight')
    plt.savefig('fig3b.svg', dpi=600, bbox_inches='tight')

    return percentage_AB, percentage_values, points_in_regions

def time_lag(ground_truth : np.array, predictions : np.array, PH : int, pred_steps : int) -> np.array:
    """
    Function that computes that time lag/delay in the CGM forecast
    like in GluNet framework [1], that is done according to [2]. 

    
    Args:
    -----
        ground_truth: array with the ground truth to be compared with the predictions
        predictions: array with the predictions of glucose values of a given model

    Returns:
    --------
        time_lag: time lag in minutes
    
    References:
    -----------
    [1] K. Li, et al., "GluNet: A Deep Learning Framework for Accurate Glucose Forecasting,"
    in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 2, pp. 414-423,
    Feb. 2020, doi: 10.1109/JBHI.2019.2931842.

    [2] C. Pérez-Gandía et al., "Artificial neural network algorithm for online glucose prediction
    from continuous glucose monitoring", Diabetes Tech. Therapeutics, vol. 12, no. 1, pp. 81-88, 2010.
    
    """

    # TBF
    time_lag = 0

    return time_lag

def model_evaluation(N : int, PH : int, name : str, normalization : str, input_features : int, 
                    X_test : np.array, Y_test : np.array, pred_steps : int, X : np.array,
                    loss_function : str, plot_results : bool = False) -> None: 
    """
    Model evaluation for a multi-step (Seq-to-seq) CGM forecasting. If the
    models are trained with normalized, in the test set the samples are denormalized
    in order to compare obtained results with those available in the  literature.
    Metrics are evaluated overall sequence and time step by time step, except for the ISO [1]
    and Parker [2] percentages that are computed only step by step.
    Evaluated metrics are: 
    - RMSE
    - MAE
    - MAPE
    - Percentage of values in the ISO 15197:2015 acceptable zone
    - Percentage of values in the Parkes Error Grid Analysis acceptable zone 

    Args:
    -----
        N: input features sequence length
        PH: prediction horizon
        name(str): name of the model
        normalization: string indicating the type of normalization applied to the data 
        input_features: number of input features
        X_test: array with the input features of the test set
        Y_test: array with the ground truth of the test set
        pred_steps: number of predicted time steps, i.e., lenght of the output sequence
        predictions: array with the predictions of glucose values of a given model
        X: array with the input features of the whole dataset (train + test) to min-max denormalize the predictions
        loss_function : str with the loss functions to load differently models trained with custom functions.
        plot_results: boolean indicating if the results must be plotted or not. Defaults to False.

        
    Returns:
    --------
        None
    
    References:
    -----------
    [1] ISO 15197:2015
    [2] Parkes
    
    """

    # If flag set to False, do not plot
    if plot_results == False:
        plt.ioff()
    else: 
        plt.ion()

    # If it is not a naive model, it is a Keras model. A different path is followed 
    if 'naive' not in name:
    
        # Load model depending on the loss function
        if loss_function == 'ISO_loss':
            model = tf.keras.models.load_model(name+'.h5', custom_objects={'ISO_adapted_loss': ISO_adapted_loss})
        else :
            model = tf.keras.models.load_model(name+'.h5', custom_objects={'ISO_adapted_loss': ISO_adapted_loss})

        # Model prediction
        Y_pred_norm = model.predict(X_test)

        # If normalization was applied, denormalize the predictions 
        if normalization == 'min-max':
            Y_pred = Y_pred_norm*(np.max(X) - np.min(X)) + np.min(X)
            X_test_denorm = X_test*(np.max(X) - np.min(X)) + np.min(X)
            Y_test_denorm = Y_test*(np.max(X) - np.min(X)) + np.min(X)
        elif normalization == None:
            Y_pred = Y_pred_norm
            X_test_denorm = X_test
            Y_test_denorm = Y_test

        # Remove second dimension of Y_pred and Y_test to compute the metrics
        Y_pred = np.squeeze(Y_pred)
        Y_test = np.squeeze(Y_test_denorm)

        # Go to previous directory 
        os.chdir('..')
    
    elif 'naive' in name:
        
        # Naive model prediction (shift the input array)
        Y_pred_norm = naive_model(X_test, input_features, round(pred_steps))
    
        # If normalization was applied, denormalize the predictions 
        if normalization == 'min-max':
            Y_pred = Y_pred_norm*(np.max(X) - np.min(X)) + np.min(X)
            X_test_denorm = X_test*(np.max(X) - np.min(X)) + np.min(X)
            Y_test_denorm = Y_test*(np.max(X) - np.min(X)) + np.min(X)
        elif normalization == None:
            Y_pred = Y_pred_norm
            X_test_denorm = X_test
            Y_test_denorm = Y_test

        # Remove second dimension of Y_pred and Y_test to compute the metrics
        Y_pred = np.squeeze(Y_pred)
        Y_test = np.squeeze(Y_test_denorm)

    # Metrics treating all the time steps at the same time 
    # Total RMSE
    rmse= np.sqrt(np.square(np.subtract(Y_test,Y_pred)).mean())
    print(name+ " Test RMSE in all time steps:  ", str(rmse))

    # MAE computation
    mae = np.mean(np.abs(Y_test-Y_pred))
    print(name+ " Test MAE in all time steps:  ", str(mae))

    # MAPE computation
    mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
    print(name+ " Test MAPE in all time steps:  ", str(mape))

    print("\n")

    # Metrics treating each time step separately
    # RMSE
    rmse = np.sqrt(np.square(np.subtract(Y_test,Y_pred)).mean(axis=0))
    print(name+ " Test RMSE in each time step:  ", str(rmse))

    # MAE
    mae = np.mean(np.abs(Y_test-Y_pred), axis=0)
    print(name+ " Test MAE in each time step:  ", str(mae))

    # MAPE
    mape = np.mean(np.abs((Y_test - Y_pred) / Y_test), axis=0) * 100
    print(name+ " Test MAPE in each time step:  ", str(mape))

    # Time lag 
    # time_lag = time_lag(Y_test, Y_pred)

    # # Go to previous directory 
    # os.chdir('..')

    # Create 'evaluation' folder is it does not exist 
    # if not os.path.exists(os.getcwd()+r"\evaluation"):
    #     os.mkdir(os.getcwd()+r"\evaluation")
    if not os.path.exists("evaluation"):
        os.mkdir("evaluation")

    # Go to the evaluation folder
    # os.chdir(os.getcwd()+r"\evaluation")
    os.chdir("evaluation")

    # Plot RMSE, MAE and MAPE for each time step
    plt.figure(figsize = (10,5))
    plt.plot(np.linspace(5,PH,round(pred_steps)), rmse, '-o', label = 'RMSE', color = 'red')
    plt.plot(np.linspace(5,PH,round(pred_steps)),mae, '-o', label = 'MAE', color = 'blue')

    # Legend
    plt.legend()

    # Set axis labels
    plt.xlabel('Time step (minutes)')
    plt.ylabel('RMSE, MAE (mg/dL)')

    # Add right axis since MAPE is %
    plt.twinx()
    plt.plot(np.linspace(5,PH,round(pred_steps)),mape, '-o', label = 'MAPE', color = 'green')
    plt.ylabel('MAPE (%)')

    # Set y limits bewteen 0 and 100
    plt.ylim(0,100)

    # Legend
    plt.legend()

    # Save the figure
    plt.savefig(name+'_multistep_metrics.png', dpi=300, bbox_inches='tight')

    # Compute the metrics once per time step
    # iso_perc, parkerAB_perc= iso_percentage_metrics(Y_test, Y_pred)

    iso_percs = []
    parkerAB_percs = []
    
    for i in range(round(pred_steps)):
        iso_perc_in, _, _ = bgISOAcceptableZone(ground_truth = Y_test[:,i], predictions = Y_pred[:,i], fold = name+'step'+str(i+1), step=i, plot=True)
        parkerAB_perc, _, _ = parkes_EGA_chart(ground_truth = Y_test[:,i], predictions = Y_pred[:,i], fold =name+'step'+str(i+1), step=i)
        iso_percs.append(iso_perc_in)
        parkerAB_percs.append(parkerAB_perc)
    
    # Plot histograms of predictions and ground truth 
    plt.figure(figsize = (10,5)) 
    plt.hist(Y_test.flatten(), bins=100, alpha=0.5)
    plt.hist(Y_pred.flatten(), bins=100, alpha=0.5)
    plt.legend(['GT', 'Prediction'])
    plt.savefig(name+'_histograms.png', dpi=300, bbox_inches='tight')

    # Save a chunk of data to plot as an example of the first predicted time step
    plt.figure(figsize = (20,10))
    plt.plot(Y_test[0:500,0], label = 'Y_test')
    plt.plot(Y_pred[0:500,0], label = 'Y_pred')
    # Set title
    plt.title('First 500 samples of the first predicted time step')
    # Legend
    plt.legend()
    # Save the plot
    plt.savefig(name+'_500samples_first_step.png', dpi=300, bbox_inches='tight')  

    # Save a chunk of data to plot as an example of the first predicted time step
    plt.figure(figsize = (20,10))
    plt.plot(Y_test[0:500,round(pred_steps)-1], label = 'Y_test')
    plt.plot(Y_pred[0:500,round(pred_steps)-1], label = 'Y_pred')
    # Set title
    plt.title('First 500 samples of the last predicted time step')
    # Legend
    plt.legend()
    # Save the plot
    plt.savefig(name+'_500samples_last_step.png', dpi=300, bbox_inches='tight') 

    # Plot 4 random predictions to visual evaluation
    random_idx = np.random.randint(0, X_test_denorm.shape[0], size = 4)

    for i in random_idx: 

    # Plot the X_test, and the difference between Y_test and Y_pred
        plt.figure(figsize = (20,10))
        plt.plot(X_test_denorm[i, :], label = 'X_test')
        plt.plot(np.linspace(N-1,N-1+round(pred_steps), round(pred_steps)), Y_test[i, :], label = 'Y_test')
        # plt.plot(np.linspace(143,149, 6), Y_pred[i, 0,:], label = 'Y_pred')
        plt.plot(np.linspace(N-1,N-1+round(pred_steps), round(pred_steps)), Y_pred[i,:], label = 'Y_pred') # for LSTM
        plt.legend()
        # Save figure
        plt.savefig(name+'_sample'+str(i)+'.png', dpi=300, bbox_inches='tight') 

    
    # "Classification" metrics separating into "hyper", "hypo" and "normal" CGM ranges 
    # First, generate the labels to compare the gorund truth to the predictions 
    gt_tags = generate_ranges_tags(Y_test)
    prediction_tags = generate_ranges_tags(Y_pred)

    # Overall accuracy
    acc = np.sum(gt_tags == prediction_tags)/len(gt_tags)

    # TP rate of different ranges  
    hypo_tp = np.sum((gt_tags == prediction_tags) & (gt_tags == 'hypo'))/np.sum(gt_tags == 'hypo')
    hyper_tp = np.sum((gt_tags == prediction_tags) & (gt_tags == 'hyper'))/np.sum(gt_tags == 'hyper')
    normal_tp = np.sum((gt_tags == prediction_tags) & (gt_tags == 'normal'))/np.sum(gt_tags == 'normal')

    # Get and save confusion matrix 
    cm = confusion_matrix(gt_tags, prediction_tags, labels = ['hypo', 'normal', 'hyper'])
    df_cm = pd.DataFrame(cm, index = ['hypo', 'normal', 'hyper'], columns = ['hypo', 'normal', 'hyper'])
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')

    # Save heatmap
    plt.savefig('confusion_matrix.png', dpi = 300)
    
    # Store results in a dictionary to further add it to the results dictionary 
    results = {'RMSE': rmse.tolist(), 'MAE': mae.tolist(), 'MAPE': mape.tolist(), 'ISO': iso_percs, 'PARKES': parkerAB_percs, 
               'Accuracy' : acc, 'Hypo TP' : hypo_tp, 'Hyper TP' : hyper_tp, 'Normal TP' : normal_tp}#, 'time_lag' : time_lag.to_list()}



    return results  

def model_evaluation_close_loop(N : int, PH : int, name : str, normalization : str,
                    X_test : np.array, Y_test : np.array, pred_steps : int, X : np.array, loss_function : str,  
                    plot_results : bool = False) -> None: 
    """
    Model evaluation for a multi-step (Seq-to-seq) CGM forecasting.
    This evaluation feeds the model with the first point of the last 
    sequence prediction, aiming a reduction of the final sequence. If the
    models are trained with normalized, in the test set the samples are denormalized
    in order to compare obtained results with those available in the  literature.
    Metrics are evaluated overall sequence and time step by time step, except for the ISO [1]
    and Parker [2] percentages that are computed only step by step.
    Evaluated metrics are: 
    - RMSE
    - MAE
    - MAPE
    - Percentage of values in the ISO 15197:2015 acceptable zone
    - Percentage of values in the Parkes Error Grid Analysis acceptable zone 

    Args:
    -----
        N: input features sequence length
        PH: prediction horizon
        name(str): name of the model
        normalization: string indicating the type of normalization applied to the data 
        X_test: array with the input features of the test set
        Y_test: array with the ground truth of the test set
        pred_steps: number of predicted time steps, i.e., lenght of the output sequence
        predictions: array with the predictions of glucose values of a given model
        X: array with the input features of the whole dataset (train + test) to min-max denormalize the predictions
        loss_function : str with the loss functions to load differently models trained with custom functions.
        plot_results: boolean indicating if the results must be plotted or not. Defaults to False.

        
    Returns:
    --------
        None
    
    References:
    -----------
    [1] ISO 15197:2015
    [2] Parkes
    
    """

    # Create 'evaluation' folder is it does not exist 
    if not os.path.exists(os.getcwd()+r"\evaluation_refeed"):
        os.mkdir(os.getcwd()+r"\evaluation_refeed")
    
    # If flag set to False, do not plot
    if plot_results == False:
        plt.ioff()
    else: 
        plt.ion()

    # Load model depending on the loss function
    if loss_function == 'ISO_loss':
        model = tf.keras.models.load_model(name+'.h5', custom_objects={'ISO_adapted_loss': ISO_adapted_loss})
    else :
        model = tf.keras.models.load_model(name+'.h5', custom_objects={'ISO_adapted_loss': ISO_adapted_loss})

    # Prediction
    Y_pred_norm = model.predict(X_test)

    # If normalization was applied, denormalize the predictions 
    if normalization == 'min-max':
        Y_pred = Y_pred_norm*(np.max(X) - np.min(X)) + np.min(X)
        X_test_denorm = X_test*(np.max(X) - np.min(X)) + np.min(X)
        Y_test_denorm = Y_test*(np.max(X) - np.min(X)) + np.min(X)
    elif normalization == None:
        Y_pred = Y_pred_norm
        X_test_denorm = X_test
        Y_test_denorm = Y_test

    # Remove second dimension of Y_pred and Y_test to compute the metrics
    Y_pred = np.squeeze(Y_pred)
    Y_test = np.squeeze(Y_test_denorm)

    # Metrics treating all the time steps at the same time 
    # Total RMSE
    rmse= np.sqrt(np.square(np.subtract(Y_test,Y_pred)).mean())
    print(name+ " Test RMSE in all time steps:  ", str(rmse))

    # MAE computation
    mae = np.mean(np.abs(Y_test-Y_pred))
    print(name+ " Test MAE in all time steps:  ", str(mae))

    # MAPE computation
    mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
    print(name+ " Test MAPE in all time steps:  ", str(mape))

    print("\n")

    # Metrics treating each time step separately
    # RMSE
    rmse = np.sqrt(np.square(np.subtract(Y_test,Y_pred)).mean(axis=0))
    print(name+ " Test RMSE in each time step:  ", str(rmse))

    # MAE
    mae = np.mean(np.abs(Y_test-Y_pred), axis=0)
    print(name+ " Test MAE in each time step:  ", str(mae))

    # MAPE
    mape = np.mean(np.abs((Y_test - Y_pred) / Y_test), axis=0) * 100
    print(name+ " Test MAPE in each time step:  ", str(mape))

    # Time lag 
    # time_lag = time_lag(Y_test, Y_pred)

    # Plot RMSE, MAE and MAPE for each time step
    plt.figure(figsize = (10,5))
    plt.plot(np.linspace(5,PH,pred_steps), rmse, '-o', label = 'RMSE', color = 'red', )
    plt.plot(np.linspace(5,PH,pred_steps),mae, '-o', label = 'MAE', color = 'blue')

    # Legend
    plt.legend()

    # Set axis labels
    plt.xlabel('Time step (minutes)')
    plt.ylabel('RMSE, MAE (mg/dL)')

    # Add right axis since MAPE is %
    plt.twinx()
    plt.plot(np.linspace(5,PH,pred_steps),mape, '-o', label = 'MAPE', color = 'green')
    plt.ylabel('MAPE (%)')

    # Set y limits bewteen 0 and 100
    plt.ylim(0,100)

    # Legend
    plt.legend()

    # Save the figure
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')

    # Go to the evaluation folder
    os.chdir(os.getcwd()+r"\evaluation_refeed")

    # Compute the metrics once per time step
    # iso_perc, parkerAB_perc= iso_percentage_metrics(Y_test, Y_pred)

    for i in range(pred_steps):
        iso_perc_in, _, _ = bgISOAcceptableZone(ground_truth = Y_test[:,i], predictions = Y_pred[:,i], fold = name+'step'+str(i+1)+'_himar-rep', step=i, plot = True)
        parkerAB_perc, _, _ = parkes_EGA_chart(ground_truth = Y_test[:,i], predictions = Y_pred[:,i], fold =name+'step'+str(i+1)+'_himar-rep', step=i)
    
    # Store results in a dictionary to further add it to the results dictionary 
    results = {'RMSE': rmse.tolist(), 'MAE': mae.tolist(), 'MAPE': mape.tolist(), 'ISO': iso_perc_in.tolist(), 'PARKES': parkerAB_perc.tolist()}#, 'time_lag' : time_lag.to_list()}

    # Plot histograms of predictions and ground truth 
    plt.figure(figsize = (10,5)) 
    plt.hist(Y_test.flatten(), bins=100, alpha=0.5)
    plt.hist(Y_pred.flatten(), bins=100, alpha=0.5)
    plt.legend(['GT', 'Prediction'])
    plt.savefig(name+'_histograms.png', dpi=300, bbox_inches='tight')

    # Save a chunk of data to plot as an example of the first predicted time step
    plt.figure(figsize = (20,10))
    plt.plot(Y_test[0:500,0], label = 'Y_test')
    plt.plot(Y_pred[0:500,0], label = 'Y_pred')
    # Set title
    plt.title('First 500 samples of the first predicted time step')
    # Legend
    plt.legend()
    # Save the plot
    plt.savefig(name+'_500samples_first_step.png', dpi=300, bbox_inches='tight')  

    # Save a chunk of data to plot as an example of the first predicted time step
    plt.figure(figsize = (20,10))
    plt.plot(Y_test[0:500,pred_steps-1], label = 'Y_test')
    plt.plot(Y_pred[0:500,pred_steps-1], label = 'Y_pred')
    # Set title
    plt.title('First 500 samples of the last predicted time step')
    # Legend
    plt.legend()
    # Save the plot
    plt.savefig(name+'_500samples_last_step.png', dpi=300, bbox_inches='tight') 

    # Plot 4 random predictions to visual evaluation
    random_idx = np.random.randint(0, X_test_denorm.shape[0], size = 4)

    for i in random_idx: 

    # Plot the X_test, and the difference between Y_test and Y_pred
        plt.figure(figsize = (20,10))
        plt.plot(X_test_denorm[i, :], label = 'X_test')
        plt.plot(np.linspace(N-1,N-1+round(PH/5), round(PH/5)), Y_test[i, :], label = 'Y_test')
        # plt.plot(np.linspace(143,149, 6), Y_pred[i, 0,:], label = 'Y_pred')
        plt.plot(np.linspace(N-1,N-1+round(PH/5), round(PH/5)), Y_pred[i,:], label = 'Y_pred') # for LSTM
        plt.legend()
        # Save figure
        plt.savefig(name+'_sample'+str(i)+'.png', dpi=300, bbox_inches='tight') 

    return results 
