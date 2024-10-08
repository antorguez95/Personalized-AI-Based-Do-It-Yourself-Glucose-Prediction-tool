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

# evaluation.py
# This module contains the functions the DL single-step models
# evaluation,including the ISO-based metrics.   
# See functions documentation for more details. 

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import tensorflow as tf

from models.training import ISO_adapted_loss


def bgISOAcceptableZone(ground_truth : np.array, predictions: np.array,  fold = str, plot : bool = False) -> Tuple[int, bool]:
    
    """
    This function generates a chart showing the ISO acceptable zone to evaluate 
    blood glucose prediction algorithms according to the ISO 15197:2015 (In vitro 
    diagnostic test systems - Requirements for blood-glucose monitoring systems for 
    self-testing in managing diabetes mellitus) [1]. It shows all predictions
    and shows also the percentage of predictions within the acceptable range (that
    must be >= 95% for ISO compliance). Since this function can be used in the
    callbacks during the Deep Learning model training, plot flag should be set to False
    to avoid memory issues with the generated figures. 

    Args:
    -----
        ground_truth: array with the ground truth to be compared with the predictions
        predictions: array with the predictions of glucose values of a given model
        fold : if many folds evaluated separately, indicate it so figures are properly saved
        plot: boolean indicating if the plot must be shown or not
        
    Returns:
    --------
        percentage_in: percentage of predicted points in the acceptable zone
        percentage_out: percentage of predicted points out of the acceptable zone
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
        plt.figure()

        # Set X and Y limits
        plt.xlim([0,550])
        plt.ylim([-90,90])

        # Set X and Y labels
        plt.xlabel('Glucose concentration (mg/dl)')
        plt.ylabel('Difference')

        # Plot boundaries in the figure
        plt.plot(region_x,regionUp_y, '--r')
        plt.plot(region_x,regionDown_y, '--r')

        # Plot error points and label 
        plt.plot(ground_truth, error,'b.')
    
        # Insert the percentage in within the acceptable range white background in the text box
        plt.text(0.05, 0.95, 'Percentage in: ' + str(round(percentage_in,2)) + '%', transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the figure
        plt.savefig(str(fold)+'_ISO15197.png', dpi=300, bbox_inches='tight')
    
    return percentage_in, percentages_out, acceptability
 
def parkes_EGA_identify(ground_truth : np.array, predictions : np.array, unit : str = "mg_dl") -> Tuple[int, int, int, int, int, int] : 
    
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

def parkes_EGA_chart(ground_truth : np.array, predictions : np.array, fold : str, unit : str = "mg_dl"):
    
    """
    Function to plot the Parkes Error Grid Analysis chart. It depends on 
    `parkes_EGA_identify` function that computes all the results. 

    Args:
    -----
        ground_truth: array with the ground truth to be compared with the predictions
        predictions: array with the predictions of glucose values of a given model  
        fold : if many folds evaluated separately, indicate it so figures are properly saved
        unit : string with the units of the glucose concentration. Default is mg/dl. Can be switch

    Returns:
    --------
        percentage_AB: percentage of values in the acceptable zone (A and B)
        percentage_values: percentage of values in each region of the Parkes Error Grid
        points_in_regions: number of points in each region of the Parkes Error Grid

    
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
    plt.figure()

    # Set X and Y limits
    plt.xlim([0,550])
    plt.ylim([0,550]) 

    # Set X an Y labels
    plt.xlabel('Glucose prediction (mg/dl)')
    plt.ylabel('Glucose reference (mg/dl)')

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

    # Save the figure
    plt.savefig(str(fold)+'_ParkerGrid.png', dpi=300, bbox_inches='tight')

    return percentage_AB, percentage_values, points_in_regions

def model_evaluation(N : int, PH : int, name : str, normalization : str, X_test : np.array, Y_test : np.array, X : np.array,
                    loss_function : str, plot_results : bool = False) -> None: 
    """
    Model evaluation for a single-step CGM forecasting. If the models are trained 
    with after min-max normalization, the samples are denormalized in the test set
    in order to compare obtained results with those available in the  literature.
    Metrics are evaluated over all sequences and time step by time step, except for
    the ISO [1] and Parker [2] percentages that are computed only step by step.
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
        normalization: string indicating the type of normalization previously applied to the data
        X_test: array with the input features of the test set
        Y_test: array with the ground truth of the test set
        X: array with the input features of the whole dataset (train + test) to min-max denormalize the predictions
        loss_function : str with the loss functions to load differently models trained with custom functions.
        plot_results: boolean indicating if the results must be plotted or not. Default is False.

        
    Returns:
    --------
        results: dictionary with the computed metrics
    
    References:
    -----------
    [1] ISO 15197:2015
    
    """

    # Create 'evaluation' folder is it does not exist 
    if not os.path.exists(os.getcwd()+r"\evaluation"):
        os.mkdir(os.getcwd()+r"\evaluation")
    
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

    # RMSE computation
    rmse= np.sqrt(np.square(np.subtract(Y_test,Y_pred)).mean())
    print(name+ " Test RMSE:  ", str(rmse))

    # MAE computation
    mae = np.mean(np.abs(Y_test-Y_pred))
    print(name+ " Test MAE:  ", str(mae))

    # MAPE computation
    mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
    print(name+ " Test MAPE:  ", str(mape))

    # Go to the evaluation folder
    os.chdir(os.getcwd()+r"\evaluation")

    iso_perc_in, b, c = bgISOAcceptableZone(ground_truth = Y_test, predictions = Y_pred, fold = 'himar-rep', plot = True)
    parkerAB_perc, b, c = parkes_EGA_chart(ground_truth = Y_test, predictions = Y_pred, fold = 'himar-rep')

    results = {'RMSE': rmse.tolist(), 'MAE': mae.tolist(), 'MAPE': mape.tolist(), 'ISO': iso_perc_in.tolist(), 'PARKES': parkerAB_perc.tolist()}#, 'time_lag' : time_lag.to_list()}

    # Plot histograms of predictions and ground truth 
    plt.figure(figsize = (10,5)) 
    plt.hist(Y_test.flatten(), bins=100, alpha=0.5)
    plt.hist(Y_pred.flatten(), bins=100, alpha=0.5)
    plt.legend(['GT', 'Prediction'])
    plt.savefig(name+'_histograms.png', dpi=300, bbox_inches='tight')
    # Plot if flag set to True

    # Save a chunk of data to plot as an example
    plt.figure(figsize = (20,10))
    plt.plot(Y_test[0:500], label = 'Y_test')
    plt.plot(Y_pred[0:500], label = 'Y_pred')
    plt.legend()
    # Save the plot
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')   
    # Plot if flag set to True

    return results 

def model_evaluation_refeed(N : int, PH : int, name : str, normalization : str, X_test : np.array, Y_test : np.array, X : np.array,
                            loss_function : str, num_predictions : int = 1000, close_loop_steps : int = 5, plot_results : bool = False) -> None: 
    
    """
    Model evaluation for a single-step CGM forecasting, evaluating the model
    with its last (predicted) output. If the models are trained after min-max normalization,
    the test set samples are denormalized in order to compare obtained results with those
    available in the  literature. Metrics are evaluated overall sequence and time step
    by time step, except for the ISO [1] and Parker [2] percentages that are computed
    only step by step. Evaluated metrics are: 
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
        X: array with the input features of the whole dataset (train + test) to min-max denormalize the predictions
        loss_function : str with the loss functions to load differently models trained with custom functions.
        num_predictions : number of predictions to be evaluated with the close loop evaluation.
        close_loop_steps : steps to give in the close loop evaluation before feed the model with real data. 
        plot_results: boolean indicating if the results must be plotted or not. Default is False.
    
    Returns:
    --------
        results: dictionary with the computed metrics
    
    References:
    -----------
    [1] ISO 15197:2015
    
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
    
    # Normal point by point prediction to compare to the close loop prediction
    # Create empty numpy 
    Y_pred_norm_normal = np.empty((0))

    # Loop to iterate over all the X test vector with jumps equal to the close loop iterations
    # for i in range(0, 1000, close_loop_iterations):
    for i in range(0, num_predictions):

        input_seq = X_test[i].reshape(1,N,1)
                        
        # Model prediction 
        pred = model.predict(input_seq)

        # Add the prediction to the Y_pred_norm vector
        Y_pred_norm_normal = np.append(Y_pred_norm_normal, pred)
    
    # Close-loop evaluation
    # Create empty numpy 
    Y_pred_norm_cl = np.empty((0))

    # Loop to iterate over all the X test vector with jumps equal to the close loop iterations
    for i in range(0, num_predictions, close_loop_steps):

        # Loop to generate one prediction at a time and make that prediction to be the last sample of the input sequence. 
        for j in range(0, close_loop_steps): 

            # When entering the loop, the input sequence is the full of the X_test vector
            if j == 0:
                input_seq = X_test[i+j].reshape(1,N,1)
            else:
                # Shift to the left the input sequence
                input_seq = np.roll(input_seq, -1, axis=1)

                # Replace last element with the prediction
                input_seq[0][N-1] = pred

                # Reshape to fit the Keras call
                input_seq = input_seq.reshape(1,N,1)
                
            # Model prediction 
            pred = model.predict(input_seq)

            print("Prev sample: ", X_test[i+j][len(X_test[i+j])-1])
            print("Prediction: ", pred)

            # Add the prediction to the Y_pred_norm vector
            Y_pred_norm_cl = np.append(Y_pred_norm_cl, pred)
    
    # If normalization was applied, denormalize the predictions 
    if normalization == 'min-max':
        Y_pred_normal = Y_pred_norm_normal*(np.max(X) - np.min(X)) + np.min(X)
        Y_pred_cl = Y_pred_norm_cl*(np.max(X) - np.min(X)) + np.min(X)
        X_test_denorm = X_test*(np.max(X) - np.min(X)) + np.min(X)
        Y_test_denorm = Y_test*(np.max(X) - np.min(X)) + np.min(X)
    elif normalization == None:
        Y_pred_normal = Y_pred_norm_normal
        Y_pred_cl = Y_pred_norm_cl
        X_test_denorm = X_test
        Y_test_denorm = Y_test

    # Remove second dimension of Y_pred and Y_test to compute the metrics
    Y_pred_normal = np.squeeze(Y_pred_normal)
    Y_pred_cl = np.squeeze(Y_pred_cl)
    Y_test = np.squeeze(Y_test_denorm)

    # RMSE computation
    rmse_normal= np.sqrt(np.square(np.subtract(Y_test,Y_pred_normal)).mean())
    rmse_cl= np.sqrt(np.square(np.subtract(Y_test,Y_pred_cl)).mean())
    print(name+ " Test RMSE (normal eval.):  ", str(rmse_normal))
    print(name+ " Test RMSE (close loop eval.):  ", str(rmse_cl))

    # MAE computation
    mae_normal = np.mean(np.abs(Y_test-Y_pred_normal))
    mae_cl = np.mean(np.abs(Y_test-Y_pred_cl))
    print(name+ " Test MAE (normal eval.):  ", str(mae_normal))
    print(name+ " Test MAE (close loop eval.):  ", str(mae_cl))

    # MAPE computation
    mape_normal= np.mean(np.abs((Y_test - Y_pred_normal) / Y_test)) * 100
    mape_cl= np.mean(np.abs((Y_test - Y_pred_cl) / Y_test)) * 100
    print(name+ " Test MAPE (normal eval.):  ", str(mape_normal))
    print(name+ " Test MAPE (close loop eval.):  ", str(mape_cl))

    # Go to the evaluation folder
    os.chdir(os.getcwd()+r"\evaluation_refeed")

    # Compute the metrics
    # iso_perc, parkerAB_perc= iso_percentage_metrics(Y_test, Y_pred)

    iso_perc_in_normal, b, c = bgISOAcceptableZone(ground_truth = Y_test, predictions = Y_pred_normal, fold = 'normal', plot = False)
    iso_perc_in_cl, b, c = bgISOAcceptableZone(ground_truth = Y_test, predictions = Y_pred_cl, fold = 'close_loop', plot = False)

    parkerAB_perc_normal, b, c = parkes_EGA_chart(ground_truth = Y_test, predictions = Y_pred_normal, fold = 'normal')
    parkerAB_perc_cl, b, c = parkes_EGA_chart(ground_truth = Y_test, predictions = Y_pred_cl, fold = 'close_loop')

    # Only plot if flag set to True
    with plt.ion():
    
        # Plot histograms of predictions and ground truth 
        plt.figure(figsize = (10,5)) 
        plt.hist(Y_test.flatten(), bins=100, alpha=0.5)
        plt.hist(Y_pred_normal.flatten(), bins=100, alpha=0.5)
        plt.hist(Y_pred_cl.flatten(), bins=100, alpha=0.5)
        plt.legend(['GT', 'Normal', 'Close loop'])
        plt.savefig(name+'_histograms.png', dpi=300, bbox_inches='tight')
        # Plot if flag set to True


        # Save a chunk of data to plot as an example
        plt.figure(figsize = (20,10))
        plt.plot(Y_test[0:500], label = 'Y_test')
        plt.plot(Y_pred_normal[0:500], label = 'Y_pred Normal')
        plt.plot(Y_pred_cl[0:500], label = 'Y_pred Close loop')
        plt.legend()
        # Save the plot
        plt.savefig(name+'.png', dpi=300, bbox_inches='tight')   
        # Plot if flag set to True

    results = {'Normal' : {'RMSE': rmse_normal.tolist(), 'MAE': mae_normal.tolist(), 'MAPE': mape_normal.tolist(), 'ISO': iso_perc_in_normal.tolist(), 'PARKES': parkerAB_perc_normal.tolist()},#, 'time_lag' : time_lag.to_list()}
               'Close loop' : {'RMSE': rmse_cl.tolist(), 'MAE': mae_cl.tolist(), 'MAPE': mape_cl.tolist(), 'ISO': iso_perc_in_cl.tolist(), 'PARKES': parkerAB_perc_cl.tolist()}}#, 'time_lag' : time_lag.to_list()}

    return results 
