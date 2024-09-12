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

import numpy as np
from typing import Tuple
# from tqdm import tqdm
import tensorflow as tf
import os
import warnings
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as KA
from tensorflow.python.client import device_lib
import time 
import pickle
import matplotlib.pyplot as plt 

# Add the parent directory to the path 
import sys
sys.path.append('..')
# from evaluation.multi_step.evaluation import bgISOAcceptableZone, parkes_EGA_chart

from typing import Tuple, Dict

from sensor_params import *

import pandas as pd
import numpy as np

# def ISO_adapted_loss(y_true: np.ndarray, y_pred: np.ndarray, n : int = 40, admisible_gamma : float = 0.1,
#                     upper_bound_error : int = 14) -> float:
def ISO_adapted_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function adapted from the classic RMSE to force the model 
    to have a larger amount of  CGM prediction points within the ISO [1] and
    Parker [2] ranges.

    Args: 
    -----
        y_true: The true glucose values.
        y_pred: The predicted glucose values.
        n : orden del comportamiento exponencial del termino de confinamiento ("filtro"). Fuera de la region de interes Default : 40.
        admisible_gamma : aportacion de de termino de confinamiento (ke2n) en la region de interes. Default : 0.1.   
        upper_bound_error : limite superior en la region de interes. Default : 14.
    
    Returns:
    --------
        loss: The loss value.

    References:
    -----------
        [1] ISO
        [2] Parker

    """

    n = tf.constant(40.0, dtype=tf.float64) # iniciamos con 40
    admisible_gamma = tf.constant(0.1, dtype=tf.float64)
    upper_bound_error = tf.constant(14.0, dtype=tf.float64)
    
    # N is the maximum between 1 and y_true/100
    N = tf.math.maximum(y_true/100, 1)

    # Error is prediction - true value
    e = y_pred - y_true

    # Normalized error
    e_norm = e/(N + KA.epsilon())

    # Squared the Error between true and predicted values
    e2 = tf.math.square(e_norm)

    # K is a constant that multiplies admisible_sigma and upper_bound_error
    K = admisible_gamma/(tf.math.pow(upper_bound_error, 2*n) + KA.epsilon())

    # Error powered 2*n
    e2n = tf.math.pow(e_norm, 2*n)

    # Final formula
    ISO_error =  e2 + K*e2n

    return ISO_error


# def iso_percentage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
#     """This computes the positive percentage of the points within the acceptable
#     range according to the ISO___ [1]. It expects the arguments to be generated
#     during the Keras fit/evaluate runtime, so first it computes the obtained
#     glucose values and then computes both percentage calling ____ for the entire batch.  

#     Args:
#     -----
#         y_true: The true glucose values.
#         y_pred: The predicted glucose values.
    
#     Returns:
#     --------
#         iso_perc: The percentage of points within the acceptable range.
#         parker_perc: The percentage of points within the acceptable range in the Parker Grid.

#     References:
#     -----------
#         [1] ISO __________
#     """
#     # If the batch size is None, return 0 for both metrics
#     if y_true.shape[0] == None:
#         return 0, 0
    
#     # for i in tqdm(range(y_true.shape[0]), desc="Computing PPV and Sens"):

#     # Compute the ISO and Parker percentages
#     iso_percentage, _ , _= bgISOAcceptableZone(y_true, y_pred)
#     parker_percentage, _, _= parkes_EGA_chart(y_true, y_pred, "_")
    
#     return iso_percentage, parker_percentage

# class CustomMetrics(tf.keras.callbacks.Callback):
#     """Custom callback to calculate the positive predicted value and sensitivity
#     metrics in the validation data after each epoch.
#     """

#     def __init__(self, data, prefix=''):
#         super(CustomMetrics, self).__init__()
#         self.data = data
#         self.prefix = prefix
#         self.mae = []
#         self.mape = []
#         # self.iso_perc = []
#         # self.parker_perc = []

#     def on_epoch_end(self, epoch, logs=None):

#         x, y = self.data
#         _mae = np.mean(np.abs(y-self.model.predict(x)), axis=0)
#         _mape = np.mean(np.abs((y - self.model.predict(x)) / y), axis=0) * 100 
#         #_iso_percentage, _parker_percentage = iso_percentage_metrics(y, self.model.predict(x))
        
#         self.mae.append(_mae)
#         self.mape.append(_mape)
#         print('- {}ISO %: {} - {}Parker %: {}'.format(self.prefix, _mae, self.prefix, _mape), end=' ')

def month_wise_4fold_cv(N : int, X: np.array, Y: np.array, X_times : np.array, training_partitions : Dict, 
                        shuffle : bool, verbose : int) -> None:
    """
    This function partitions the data in 4 folds, so models are trained with all seasons and validated with 
    all season: 

    Fold 1: January - September (train) and October - December (test)  
    Fold 2: January - June (train) and July - September (test)
    Fold 3: January - March (train) and April - June (test)
    Fold 4: April - December (train) and January - March (test)

    Dara is stored in its correspondant fold in the dictionary training_partitions.

    Args:
    -----
    N: window size of the input data.
    X: input sequence of lenght N.
    Y: output sequence.
    X_times: time stamps of the input sequence.
    training_partitions: dictionary where the data will be stored.
    shuffle: flag that indicates whether to shuffle the data or not.
    verbose: verbosity level. 

    Returns:
    --------
    None

    """
 
    # Unique entry for this training strategy in the dictionary 
    dict_entry = "month_wise_4fold_cv"
    
    # If training partitions do not have an entry called dict_entry create it
    if dict_entry not in training_partitions.keys(): 
        training_partitions[dict_entry] = {}


        # Add keys correspondant to the years and partitions and the final data set
        years = ['2020', '2021']
        time_boundaries = ['first_january', 'first_april', 'first_july', 'first_october', 
                        'last_march', 'last_june', 'last_september', 'last_december']
        folds = ['1','2','3', '4']
        fold_keys = ['1-fold', '2-fold', '3-fold', '4-fold']

        for i in range(0, len(years)):
            training_partitions[dict_entry][years[i]] = {'time_boundaries' : {}, 'folds' : {}}
            
            # Set boundaries keys
            for j in range(0, len(time_boundaries)):
                training_partitions[dict_entry][years[i]]['time_boundaries'][time_boundaries[j]] = {}
            
            # Set folds keys
            for j in range(0, len(folds)): 
                training_partitions[dict_entry][years[i]]['folds'][folds[j]] = {}
                
        # Declaration of the time boundaries for 2020 and store in dictionary  
        # !!!!!!!! PENDING TO CHANGE FOR SECONDS UNIT !!!!!!!!!!!!!
        training_partitions[dict_entry]['2020']['time_boundaries']['first_january'] = pd.to_datetime('2020-01-01 00:00:00')
        training_partitions[dict_entry]['2020']['time_boundaries']['first_april'] = pd.to_datetime('2020-04-01 00:00:00')
        training_partitions[dict_entry]['2020']['time_boundaries']['first_july'] = pd.to_datetime('2020-07-01 00:00:00')
        training_partitions[dict_entry]['2020']['time_boundaries']['first_october'] = pd.to_datetime('2020-10-01 00:00:00')

        training_partitions[dict_entry]['2020']['time_boundaries']['last_march'] = pd.to_datetime('2020-03-31 23:59:59')
        training_partitions[dict_entry]['2020']['time_boundaries']['last_june'] = pd.to_datetime('2020-06-30 23:59:59')
        training_partitions[dict_entry]['2020']['time_boundaries']['last_september'] = pd.to_datetime('2020-09-30 23:59:59')
        training_partitions[dict_entry]['2020']['time_boundaries']['last_december'] = pd.to_datetime('2020-12-31 23:59:59')

        training_partitions[dict_entry]['2021']['time_boundaries']['first_january'] = pd.to_datetime('2021-01-01 00:00:00')
        training_partitions[dict_entry]['2021']['time_boundaries']['first_april'] = pd.to_datetime('2021-04-01 00:00:00')
        training_partitions[dict_entry]['2021']['time_boundaries']['first_july'] = pd.to_datetime('2021-07-01 00:00:00')
        training_partitions[dict_entry]['2021']['time_boundaries']['first_october'] = pd.to_datetime('2021-10-01 00:00:00')

        training_partitions[dict_entry]['2021']['time_boundaries']['last_march'] = pd.to_datetime('2021-03-31 23:59:59')
        training_partitions[dict_entry]['2021']['time_boundaries']['last_june'] = pd.to_datetime('2021-06-30 23:59:59')
        training_partitions[dict_entry]['2021']['time_boundaries']['last_september'] = pd.to_datetime('2021-09-30 23:59:59')
        training_partitions[dict_entry]['2021']['time_boundaries']['last_december'] = pd.to_datetime('2021-12-31 23:59:59')

        # 2020 and 2021 month-wise data partition 
        X_2020 = X[np.where((X_times[:,0] > pd.to_datetime('2020-01-01 00:00:00')) & (X_times[:,N-1] < pd.to_datetime('2020-12-31 23:59:59')))[0]]
        X_times_2020 = X_times[np.where((X_times[:,0] > pd.to_datetime('2020-01-01 00:00:00')) & (X_times[:,N-1] < pd.to_datetime('2020-12-31 23:59:59')))[0]]

        X_2021 = X[np.where((X_times[:,0] > pd.to_datetime('2021-01-01 00:00:00')) & (X_times[:,N-1] < pd.to_datetime('2021-12-31 23:59:59')))[0]]
        X_times_2021 = X_times[np.where((X_times[:,0] > pd.to_datetime('2021-01-01 00:00:00')) & (X_times[:,0] < pd.to_datetime('2021-12-31 23:59:59')))[0]]

        # Iterate over the the different years to obtain the subdatasets
        for year in training_partitions[dict_entry].keys():

            print("Obtaining folds for year "+year+" ...\n ")
        
            # 1st fold
            training_partitions[dict_entry][year]['folds']['1']['X_train'] = X[np.where(((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january']) 
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_september'])))[0]]
            training_partitions[dict_entry][year]['folds']['1']['X_test'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_october']) 
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december']))[0]]
            training_partitions[dict_entry][year]['folds']['1']['Y_train'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january']) 
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_september']))[0]]
            training_partitions[dict_entry][year]['folds']['1']['Y_test'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_october']) 
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december']))[0]]
            
            # 2nd fold
            training_partitions[dict_entry][year]['folds']['2']['X_train'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january']) 
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_june'])
                                                                                | ((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_october']) 
                                                                                & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december'])))[0]]
            training_partitions[dict_entry][year]['folds']['2']['X_test'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_july'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_september']))[0]]
            training_partitions[dict_entry][year]['folds']['2']['Y_train'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_june'])
                                                                                | ((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_october']) 
                                                                                    & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december'])))[0]]
            training_partitions[dict_entry][year]['folds']['2']['Y_test'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_july']) 
                                                                        & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_september']))[0]]
        
            # 3rd fold
            training_partitions[dict_entry][year]['folds']['3']['X_train'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_march'])
                                                                            | ((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_july'])
                                                                                & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december'])))[0]]
            training_partitions[dict_entry][year]['folds']['3']['X_test'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_april'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_june']))[0]]
            training_partitions[dict_entry][year]['folds']['3']['Y_train']  = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_march'])
                                                                                | ((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_july'])
                                                                                    & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december'])))[0]]
            training_partitions[dict_entry][year]['folds']['3']['Y_test'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_april'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_june']))[0]]
            
            # 4th fold
            training_partitions[dict_entry][year]['folds']['4']['X_train'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_april'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december']))[0]]
            training_partitions[dict_entry][year]['folds']['4']['X_test'] = X[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_march']))[0]]
            training_partitions[dict_entry][year]['folds']['4']['Y_train'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_april'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_december']))[0]]
            training_partitions[dict_entry][year]['folds']['4']['Y_test'] = Y[np.where((X_times[:,0] >= training_partitions[dict_entry][year]['time_boundaries']['first_january'])
                                                                            & (X_times[:,N-1] <= training_partitions[dict_entry][year]['time_boundaries']['last_march']))[0]]
        if verbose > 0:

            # Print dimensions of  X_train, X_test, Y_train, Y_test, and the difference respected to their sum and the original X and Y
            for fold in training_partitions[dict_entry]['2020']['folds']:
                print("--------------- 2020 - 1st fold ---------------")
                print("Train instances: ", training_partitions[dict_entry]['2020']['folds'][fold]['X_train'].shape[0])
                print("Test instances ", training_partitions[dict_entry]['2020']['folds'][fold]['X_test'].shape[0])
                print("Discarded samples due to the presence of 2 different months in one patient: ", X_2020.shape[0] - (training_partitions[dict_entry]['2020']['folds'][fold]['X_train'].shape[0] + training_partitions[dict_entry]['2020']['folds'][fold]['X_test'].shape[0]))

                print("\n----------------------------------------------------------\n")

        if verbose > 0: 
        
            for fold in training_partitions[dict_entry]['2021']['folds']:
                print("--------------- 2021 - 1st fold ---------------")
                print("Train instances: ", training_partitions[dict_entry]['2021']['folds'][fold]['X_train'].shape[0])
                print("Test instances ", training_partitions[dict_entry]['2021']['folds'][fold]['X_test'].shape[0])
                print("Discarded samples due to the presence of 2 different months in one patient: ", X_2021.shape[0] - (training_partitions[dict_entry]['2021']['folds'][fold]['X_train'].shape[0] + training_partitions[dict_entry]['2021']['folds'][fold]['X_test'].shape[0]))

                print("\n----------------------------------------------------------\n")

        # Concatenate X and Y for shuffling without losing the correspondence between vectors
        for year in training_partitions[dict_entry].keys():
            for fold in training_partitions[dict_entry][year]['folds']:
                training_partitions[dict_entry][year]['folds'][fold]['XY_train'] = np.concatenate((training_partitions[dict_entry][year]['folds'][fold]['X_train'], training_partitions[dict_entry][year]['folds'][fold]['Y_train']), axis=1)
                training_partitions[dict_entry][year]['folds'][fold]['XY_test'] = np.concatenate((training_partitions[dict_entry][year]['folds'][fold]['X_test'], training_partitions[dict_entry][year]['folds'][fold]['Y_test']), axis=1)


        # Joint 2020 and 2021 folds
        XY_train_fold1 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['1']['XY_train'], training_partitions[dict_entry]['2021']['folds']['1']['XY_train']), axis=0)
        XY_test_fold1 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['1']['XY_test'], training_partitions[dict_entry]['2021']['folds']['1']['XY_test']), axis=0)

        XY_train_fold2 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['2']['XY_train'], training_partitions[dict_entry]['2021']['folds']['2']['XY_train']), axis=0)
        XY_test_fold2 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['2']['XY_test'], training_partitions[dict_entry]['2021']['folds']['2']['XY_test']), axis=0)

        XY_train_fold3 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['3']['XY_train'], training_partitions[dict_entry]['2021']['folds']['3']['XY_train']), axis=0)
        XY_test_fold3 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['3']['XY_test'], training_partitions[dict_entry]['2021']['folds']['3']['XY_test']), axis=0)

        XY_train_fold4 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['4']['XY_train'], training_partitions[dict_entry]['2021']['folds']['4']['XY_train']), axis=0)
        XY_test_fold4 = np.concatenate((training_partitions[dict_entry]['2020']['folds']['4']['XY_test'], training_partitions[dict_entry]['2021']['folds']['4']['XY_test']), axis=0)


        # Shuffle the data if required
        if shuffle == 1:
            np.random.shuffle(XY_train_fold1)
            np.random.shuffle(XY_train_fold2)
            np.random.shuffle(XY_train_fold3)
            np.random.shuffle(XY_train_fold4)

            print("Data shuffled\n")
        else: 
            print("Data not shuffled\n")

        # Entry to store the final X and Y training and test subsets (not done before to iterate only over keys
        # that correspond to years)
        training_partitions[dict_entry]['partitioned_dataset'] = {}

        for i in range(0, len(folds)):
            training_partitions[dict_entry]['partitioned_dataset'][fold_keys[i]] = {'X_train' : {},
                                                                                'X_test' : {},  
                                                                                'Y_train' : {},
                                                                                'Y_test' : {}}
        
        train_folds = [XY_train_fold1, XY_train_fold2, XY_train_fold3, XY_train_fold4]
        test_folds = [XY_test_fold1, XY_test_fold2, XY_test_fold3, XY_test_fold4]

        # Separate X and Y again
        for i in range(0, len(train_folds)):
            training_partitions[dict_entry]['partitioned_dataset'][fold_keys[i]]['X_train'] = train_folds[i][:,0:N]
            training_partitions[dict_entry]['partitioned_dataset'][fold_keys[i]]['Y_train'] = train_folds[i][:,N:]
            training_partitions[dict_entry]['partitioned_dataset'][fold_keys[i]]['X_test'] = test_folds[i][:,0:N]
            training_partitions[dict_entry]['partitioned_dataset'][fold_keys[i]]['Y_test'] = test_folds[i][:,N:] 

    else: 
        # We assume that if the key exists, so does the partition
        print("Month-wise 4-fold Cross Validation partition already done.") 

def month_wise_LibreView_4fold_cv(X: np.array, Y: np.array, X_times : np.array, Y_times : np.array, levels_tags : np.array, N: int) -> Dict:

    """
    This function partitions the data in 4 folds. Each fold contains data from 3 months of the same year.
    With this, each model is trained and validated with all different terms in a year. The timestamps 
    of the folds will vary depending on the patient. The oldest recorded sample in the patient will be the 
    first sample of the first fold. The first sample of the second fold will be that sample plus 3 months,
    and so on. This function has been designed to work with LibreView-extracted data, but can be adapted to 
    other data sources. Information about the partitions is stored in a .txt file.

    Data is stored in its correspondant fold in the dictionary training_partitions.

    Args:
    -----
        X: input sequence of lenght N.
        Y: output sequence.
        X_times: timestamps of the input sequence.
        Y_times: timestamps of the output sequence.
        levels_tags: array with the tag ("hyper", "hypo", "normal") of each sample considering the Y sequence (prediction).        
        N: window size of the input data.
        shuffle: flag that indicates whether to shuffle the data or not.
        verbose: verbosity level. 

    Returns:
    --------
        folds_dict: dictionary containing the 4 folds. Each fold contains the training and validation sets.
    

    """

    # Declare the dictionary to intuitively access the folds 
    folds_dict = {'1-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'X_test' : {},
                            'Y_test' : {}},
                '2-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'X_test' : {},
                            'Y_test' : {}},            
                '3-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'X_test' : {},
                            'Y_test' : {}},
                '4-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'X_test' : {},
                            'Y_test' : {}}}
    
    # Timestamp of the fold 1 is the first of the whole recording 
    fold1_first_timestamp = X_times[0][0]

    # Timestamp of the fold 2 is the first of the whole recording + 3 months
    fold2_first_timestamp = fold1_first_timestamp + pd.DateOffset(months=3)

    # Timestamp of the fold 3 is the first of the whole recording + 6 months
    fold3_first_timestamp = fold1_first_timestamp + pd.DateOffset(months=6)

    # Timestamp of the fold 4 is the first of the whole recording + 9 months
    fold4_first_timestamp = fold1_first_timestamp + pd.DateOffset(months=9)

    # With the timestamps, the 4 folds are generated
    X_fold1 = X[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    X_fold2 = X[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    X_fold3 = X[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    X_fold4 = X[np.where(X_times[:,0] >= fold4_first_timestamp)] 

    # Also save the timestamps of the fold just in case they are necessary 
    X_times_fold1 = X_times[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    X_times_fold2 = X_times[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    X_times_fold3 = X_times[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    X_times_fold4 = X_times[np.where(X_times[:,0] >= fold4_first_timestamp)]

    # Take the same instances from Y
    Y_fold1 = Y[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    Y_fold2 = Y[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    Y_fold3 = Y[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    Y_fold4 = Y[np.where(X_times[:,0] >= fold4_first_timestamp)]

    # Take the same instances from Y_times
    Y_times_fold1 = Y_times[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    Y_times_fold2 = Y_times[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    Y_times_fold3 = Y_times[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    Y_times_fold4 = Y_times[np.where(X_times[:,0] >= fold4_first_timestamp)]

    lost_samples = len(X) - (len(X_fold1) + len(X_fold2) + len(X_fold3) + len(X_fold4))

    print("Discarded instances: %i" % (lost_samples))

    # Save valuable information in a .txt file
    with open('4-folds_summary.txt', 'w') as f:
        f.write('1-fold start date = {}\n'.format(fold1_first_timestamp))
        f.write('1-fold num. samples = {}\n\n'.format(len(X_fold1)))

        f.write('2-fold start date = {}\n'.format(fold2_first_timestamp))
        f.write('2-fold num. samples = {}\n\n'.format(len(X_fold2)))

        f.write('3-fold start date = {}\n'.format(fold3_first_timestamp))
        f.write('3-fold num. samples = {}\n\n'.format(len(X_fold3)))

        f.write('4-fold start date = {}\n'.format(fold4_first_timestamp))
        f.write('4-fold num. samples = {}\n\n'.format(len(X_fold4)))

        f.write('Discarded instances due to overlap = {}\n'.format(lost_samples))

    # Concatenate XY in the same array but in a different axis. Just once to shuflle later 
    XY_fold1 = np.concatenate((X_fold1, Y_fold1), axis=1)
    XY_fold2 = np.concatenate((X_fold2, Y_fold2), axis=1)
    XY_fold3 = np.concatenate((X_fold3, Y_fold3), axis=1)
    XY_fold4 = np.concatenate((X_fold4, Y_fold4), axis=1)

    # Create the training sets for each fold 
    fold1_XY_train_set = np.concatenate((XY_fold1, XY_fold2, XY_fold3), axis=0)
    fold2_XY_train_set = np.concatenate((XY_fold1, XY_fold2, XY_fold4), axis=0)
    fold3_XY_train_set = np.concatenate((XY_fold1, XY_fold3, XY_fold4), axis=0)
    fold4_XY_train_set = np.concatenate((XY_fold2, XY_fold3, XY_fold4), axis=0)

    # Shuffle the training sets
    np.random.shuffle(fold1_XY_train_set)
    np.random.shuffle(fold2_XY_train_set)
    np.random.shuffle(fold3_XY_train_set)
    np.random.shuffle(fold4_XY_train_set)

    # Split the training sets into X and Y
    fold1_X_train = fold1_XY_train_set[:,0:N]
    fold1_Y_train = fold1_XY_train_set[:,N:]

    fold2_X_train = fold2_XY_train_set[:,0:N]
    fold2_Y_train = fold2_XY_train_set[:,N:]

    fold3_X_train = fold3_XY_train_set[:,0:N]
    fold3_Y_train = fold3_XY_train_set[:,N:]

    fold4_X_train = fold4_XY_train_set[:,0:N]
    fold4_Y_train = fold4_XY_train_set[:,N:]

    # Fill the dictionary fold-wise
    # 1-fold
    folds_dict['1-fold']['X_train'] = fold1_X_train
    folds_dict['1-fold']['Y_train'] = fold1_Y_train
    folds_dict['1-fold']['X_test'] = X_fold4
    folds_dict['1-fold']['Y_test'] = Y_fold4

    # 2-fold
    folds_dict['2-fold']['X_train'] = fold2_X_train
    folds_dict['2-fold']['Y_train'] = fold2_Y_train
    folds_dict['2-fold']['X_test'] = X_fold3
    folds_dict['2-fold']['Y_test'] = Y_fold3

    # 3-fold
    folds_dict['3-fold']['X_train'] = fold3_X_train
    folds_dict['3-fold']['Y_train'] = fold3_Y_train
    folds_dict['3-fold']['X_test'] = X_fold2
    folds_dict['3-fold']['Y_test'] = Y_fold2

    # 4-fold
    folds_dict['4-fold']['X_train'] = fold4_X_train
    folds_dict['4-fold']['Y_train'] = fold4_Y_train
    folds_dict['4-fold']['X_test'] = X_fold1
    folds_dict['4-fold']['Y_test'] = Y_fold1

    return folds_dict 

def train_model(sensor : Dict, 
                model : tf.keras.Model,
                X: np.array, 
                Y: np.array,
                N: int, 
                predicted_points: int, 
                epochs: int, 
                batch_size: int, 
                lr: float,
                fold : int,
                sample_weights : np.array = None,
                loss_function : str = 'root_mean_squared_error',
                verbose : int = 1,
                plot : bool = False) -> None: 
    
    """Train a previously loaded Deep Learning model using 
    the given data, and some model hyperparameters. 

    Args:
    -----
        model (tf.keras.Model): The model to train.
        X (np.array): The input features (size = N).
        Y (np.array): The output sequence (size = predicted_points).
        N (int): Input features length.
        predicted_points (int): Number of points to predict, i.e., the output sequence length.
        epochs (int): Number of epochs.
        sensor (Dict): Dictionary with the sensor name and its parameters.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        fold (int): When training with cross-validation, the fold to train and save the model with its name. 
        loss_function (str): Loss function to use. Defaults to 'root_mean_squared_error'.
        verbose (int): Verbosity level. Defaults to 1.

    Returns:
    --------
        None
    """
    
    # Avoid plotting 
    if plot == False: 
        plt.ioff()
    else: 
        plt.ion()
    
    # Compile model dependin on the loss function (for now it will be the same)
    if loss_function == 'root_mean_squared_error':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.keras.losses.MeanSquaredError(), 
                    metrics=[tf.keras.metrics.RootMeanSquaredError(), ISO_adapted_loss]) 
    
    elif loss_function == 'ISO_loss':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=ISO_adapted_loss, 
                    metrics=[tf.keras.metrics.RootMeanSquaredError(), ISO_adapted_loss])
    else :
        raise ValueError('The loss function {} is not implemented'.format(loss_function))
    
    # Modular path of the model. Current directory assumed to be the parent directory 
    dir = os.getcwd()
    
    # Create directory if it has not been previously created
    if 'training' not in dir:  
        # Create directory to store the training parameters and results
        # training_path = r'\training'
        training_path = '/training'
        if not os.path.exists(dir+training_path):
            os.makedirs(dir+training_path)
        # Change to that directory 
        os.chdir(dir+training_path)
        dir = dir+training_path
    else: 
        pass
    
    # Save model summary in a txt file
    filename = str(fold)+'_modelsummary.txt'
    with open('modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save training configuration in a txt file
    with open('training_configuration.txt', 'w') as f:
        f.write('N = {}\n'.format(N))
        f.write('epochs = {}\n'.format(epochs))
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('lr = {}\n'.format(lr))

    # Save model plot in a png file
    try:
        plot_model(model, to_file=dir+'/model_plot.png')
    except(ImportError):
        if verbose >=2:
            warnings.warn('Could not plot the model. Install plot_model dependencies to that.', UserWarning)
        pass

    # Annote the initial time
    t0 = time.time()

    # Define delta for early stopping depending on the loss function
    if loss_function == 'root_mean_squared_error':
        #delta = 0.05
        delta = 0.0001
    
    elif loss_function == 'ISO_loss':
        delta = 0.0001

    else :
        raise ValueError('The loss function {} is not implemented'.format(loss_function))
    
    # Callback to implement early stopping
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', # monitor always the loss function
                                                min_delta = delta,
                                                patience=2, # epochs to wait until stop
                                                mode = "min",
                                                restore_best_weights=True,
                                                )
    
    print('Running training using  GPU: {}'.format([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']))
    
    # Model training
    history = model.fit(
            x=X,
            y=Y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks= callbacks,
            sample_weight=sample_weights,
            verbose=verbose
        )

    # Calculate the time elapsed
    t1 = time.time() - t0

    # Add the time elapsed to the training txt
    filename = str(fold)+'_training.txt'
    with open(filename, 'a') as f:
        f.write('time elapsed: {} s'.format(t1))

    # Save history object
    with open(str(fold)+'_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot training and validation loss
    fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,5))
    axs[0].plot(history.history['loss'], label='Training')
    axs[0].set_ylabel(loss_function)

    axs[0].legend()

    # Plot the other metric different from the loss function to evaluate 
    if loss_function == 'root_mean_squared_error':
        axs[1].plot(history.history['ISO_adapted_loss'], label='Training')
        axs[1].set_ylabel('ISO_loss')
    elif loss_function == 'ISO_loss':
        axs[1].plot(history.history['root_mean_squared_error'], label='Training')
        axs[1].set_ylabel('RMSE')

    axs[1].set_xlabel('Epoch')

    # Reduce the space between the subplots
    fig.subplots_adjust(hspace=0)

    plt.suptitle(str(fold)+' Training evolution: N = {}, PH = {} min.' .format(N, predicted_points*sensor['SAMPLE_PERIOD']))

    # Save the figure
    plt.savefig(dir+'/'+str(fold)+'_training_evolution.png')
    # plt.savefig(str(fold)+'_training_evolution.png')

    # Model name (depends on the fold to train)
    model_name = '/'+str(fold) +'.h5'
    
    # Save the model
    model.save(dir+model_name)
    # model.save(model_name)

    if verbose >= 1:
        print('\n\tEnd of the training. Model saved in {}\n'.format(dir))
    
def month_wise_multi_input_LibreView_4fold_cv(X: np.array, Y: np.array, X_times : np.array, Y_times : np.array, levels_tags: np.array,
                                            N: int, input_features : int) -> Dict:

    """
    This function partitions the multi input data in 4 folds. Each fold contains data from 3 months of the same year.
    With this, each model is trained and validated with all different terms in a year. The timestamps 
    of the folds will vary depending on the patient. The oldest recorded sample in the patient will be the 
    first sample of the first fold. The first sample of the second fold will be that sample plus 3 months,
    and so on. This function has been designed to work with LibreView-extracted data, but can be adapted to 
    other data sources. Information about the partitions is stored in a .txt file.

    Data is stored in its correspondant fold in the dictionary training_partitions.

    Args:
    -----
        X: input sequence of lenght N (multi input).
        Y: output sequence.
        X_times: timestamps of the input sequence.
        Y_times: timestamps of the output sequence.
        levels_tags: array with the tag ("hyper", "hypo", "normal") of each sample considering the Y sequence (prediction).
        N: window size of the input data.
        input_features: number of input features.

    Returns:
    --------
        folds_dict: dictionary containing the 4 folds. Each fold contains the training and validation sets.
    

    """

    folds_dict = {'1-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'train_tags' : {},
                            'X_test' : {},
                            'Y_test' : {},
                            'test_tags' : {}},
                '2-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'train_tags' : {},
                            'X_test' : {},
                            'Y_test' : {},
                            'test_tags' : {}},            
                '3-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'train_tags' : {},
                            'X_test' : {},
                            'Y_test' : {},
                            'test_tags' : {}},
                '4-fold' : {'X_train' : {},
                            'Y_train' : {},
                            'train_tags' : {},
                            'X_test' : {},
                            'Y_test' : {}, 
                            'test_tags' : {}}}

    # Timestamp of the fold 1 is the first of the whole recording 
    fold1_first_timestamp = X_times[0][0]

    # Timestamp of the fold 2 is the first of the whole recording + 3 months
    fold2_first_timestamp = fold1_first_timestamp + pd.DateOffset(months=3)

    # Timestamp of the fold 3 is the first of the whole recording + 6 months
    fold3_first_timestamp = fold1_first_timestamp + pd.DateOffset(months=6)

    # Timestamp of the fold 4 is the first of the whole recording + 9 months
    fold4_first_timestamp = fold1_first_timestamp + pd.DateOffset(months=9)

    # With the timestamps, the 4 folds are generated
    X_fold1 = X[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    X_fold2 = X[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    X_fold3 = X[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    X_fold4 = X[np.where(X_times[:,0] >= fold4_first_timestamp)] 

    # Also save the timestamps of the fold just in case they are necessary 
    X_times_fold1 = X_times[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    X_times_fold2 = X_times[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    X_times_fold3 = X_times[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    X_times_fold4 = X_times[np.where(X_times[:,0] >= fold4_first_timestamp)]

    # Take the same instances from Y
    Y_fold1 = Y[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    Y_fold2 = Y[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    Y_fold3 = Y[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    Y_fold4 = Y[np.where(X_times[:,0] >= fold4_first_timestamp)]

    # Take the same instances from Y_times
    Y_times_fold1 = Y_times[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    Y_times_fold2 = Y_times[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    Y_times_fold3 = Y_times[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    Y_times_fold4 = Y_times[np.where(X_times[:,0] >= fold4_first_timestamp)]

    # Take the same instances from levels_tags
    levels_tags_fold1 = levels_tags[np.where((X_times[:,0] >= fold1_first_timestamp) & (Y_times[:,0] < fold2_first_timestamp))[0]]
    levels_tags_fold2 = levels_tags[np.where((X_times[:,0] >= fold2_first_timestamp) & (Y_times[:,0] < fold3_first_timestamp))[0]]
    levels_tags_fold3 = levels_tags[np.where((X_times[:,0] >= fold3_first_timestamp) & (Y_times[:,0] < fold4_first_timestamp))[0]]
    levels_tags_fold4 = levels_tags[np.where(X_times[:,0] >= fold4_first_timestamp)]

    lost_samples = len(X) - (len(X_fold1) + len(X_fold2) + len(X_fold3) + len(X_fold4))

    print("Discarded instances: %i" % (lost_samples))

    # Save valuable information in a .txt file
    with open('4-folds_summary.txt', 'w') as f:
        f.write('1-fold start date = {}\n'.format(fold1_first_timestamp))
        f.write('1-fold num. samples = {}\n\n'.format(len(X_fold1)))

        f.write('2-fold start date = {}\n'.format(fold2_first_timestamp))
        f.write('2-fold num. samples = {}\n\n'.format(len(X_fold2)))

        f.write('3-fold start date = {}\n'.format(fold3_first_timestamp))
        f.write('3-fold num. samples = {}\n\n'.format(len(X_fold3)))

        f.write('4-fold start date = {}\n'.format(fold4_first_timestamp))
        f.write('4-fold num. samples = {}\n\n'.format(len(X_fold4)))

        f.write('Discarded instances due to overlap = {}\n'.format(lost_samples))

    # Concatenate XY in the same array but in a different axis. Just once to shuflle later 
    # Reshape Y to 3D
    Y_fold1 = np.reshape(Y_fold1, (Y_fold1.shape[0], Y_fold1.shape[1], 1))
    Y_fold2 = np.reshape(Y_fold2, (Y_fold2.shape[0], Y_fold2.shape[1], 1))
    Y_fold3 = np.reshape(Y_fold3, (Y_fold3.shape[0], Y_fold3.shape[1], 1))
    Y_fold4 = np.reshape(Y_fold4, (Y_fold4.shape[0], Y_fold4.shape[1], 1))

    # Reshape levels_tags to 3D
    levels_tags_fold1 = np.reshape(levels_tags_fold1, (levels_tags_fold1.shape[0], levels_tags_fold1.shape[1], 1))
    levels_tags_fold2 = np.reshape(levels_tags_fold2, (levels_tags_fold2.shape[0], levels_tags_fold2.shape[1], 1))
    levels_tags_fold3 = np.reshape(levels_tags_fold3, (levels_tags_fold3.shape[0], levels_tags_fold3.shape[1], 1))
    levels_tags_fold4 = np.reshape(levels_tags_fold4, (levels_tags_fold4.shape[0], levels_tags_fold4.shape[1], 1))

    # Y_fold1_concat is a copy of Y_fold1
    Y_fold1_concat = Y_fold1
    Y_fold2_concat = Y_fold2
    Y_fold3_concat = Y_fold3
    Y_fold4_concat = Y_fold4

    # For the sake of concatenation, add a replication of Y_fold and level tags the times needed to match the number of input features 
    for i in range(0, input_features-1):
        Y_fold1_concat = np.concatenate((Y_fold1_concat, Y_fold1), axis=2)
        Y_fold2_concat = np.concatenate((Y_fold2_concat, Y_fold2), axis=2)
        Y_fold3_concat = np.concatenate((Y_fold3_concat, Y_fold3), axis=2)
        Y_fold4_concat = np.concatenate((Y_fold4_concat, Y_fold4), axis=2)

        levels_tags_fold1 = np.concatenate((levels_tags_fold1, levels_tags_fold1), axis=2)
        levels_tags_fold2 = np.concatenate((levels_tags_fold2, levels_tags_fold2), axis=2)
        levels_tags_fold3 = np.concatenate((levels_tags_fold3, levels_tags_fold3), axis=2)
        levels_tags_fold4 = np.concatenate((levels_tags_fold4, levels_tags_fold4), axis=2)

    # Concatenate XY in the same array but in a different axis. Just once to shuflle later 
    XY_fold1 = np.concatenate((X_fold1, Y_fold1_concat), axis=1)
    XY_fold2 = np.concatenate((X_fold2, Y_fold2_concat), axis=1)
    XY_fold3 = np.concatenate((X_fold3, Y_fold3_concat), axis=1)
    XY_fold4 = np.concatenate((X_fold4, Y_fold4_concat), axis=1)

    # Concatenate XY with the level tags 
    XY_fold1 = np.concatenate((XY_fold1, levels_tags_fold1), axis=1)
    XY_fold2 = np.concatenate((XY_fold2, levels_tags_fold2), axis=1)
    XY_fold3 = np.concatenate((XY_fold3, levels_tags_fold3), axis=1)
    XY_fold4 = np.concatenate((XY_fold4, levels_tags_fold4), axis=1)

    # Create the training sets for each fold 
    fold1_XY_train_set = np.concatenate((XY_fold1, XY_fold2, XY_fold3), axis=0)
    fold2_XY_train_set = np.concatenate((XY_fold1, XY_fold2, XY_fold4), axis=0)
    fold3_XY_train_set = np.concatenate((XY_fold1, XY_fold3, XY_fold4), axis=0)
    fold4_XY_train_set = np.concatenate((XY_fold2, XY_fold3, XY_fold4), axis=0)

    # Shuffle the training sets
    np.random.shuffle(fold1_XY_train_set)
    np.random.shuffle(fold2_XY_train_set)
    np.random.shuffle(fold3_XY_train_set)
    np.random.shuffle(fold4_XY_train_set)

    # Split the training sets into X, Y and tags
    fold1_X_train = fold1_XY_train_set[:,0:N]
    fold1_Y_train = fold1_XY_train_set[:,N:N+Y_fold1.shape[1]]
    fold1_tags_train = fold1_XY_train_set[:,N+Y_fold1.shape[1]:]

    fold2_X_train = fold2_XY_train_set[:,0:N]
    fold2_Y_train = fold2_XY_train_set[:,N:N+Y_fold1.shape[1]]
    fold2_tags_train = fold2_XY_train_set[:,N+Y_fold1.shape[1]:]

    fold3_X_train = fold3_XY_train_set[:,0:N]
    fold3_Y_train = fold3_XY_train_set[:,N:N+Y_fold1.shape[1]]
    fold3_tags_train = fold3_XY_train_set[:,N+Y_fold1.shape[1]:]

    fold4_X_train = fold4_XY_train_set[:,0:N]
    fold4_Y_train = fold4_XY_train_set[:,N:N+Y_fold1.shape[1]]
    fold4_tags_train = fold4_XY_train_set[:,N+Y_fold1.shape[1]:]

    # Drop the additional dimensions of Y and level tags 
    fold1_Y_train = fold1_Y_train[:,:,0]
    fold2_Y_train = fold2_Y_train[:,:,0]
    fold3_Y_train = fold3_Y_train[:,:,0]
    fold4_Y_train = fold4_Y_train[:,:,0]

    fold1_tags_train = fold1_tags_train[:,:,0]
    fold2_tags_train = fold2_tags_train[:,:,0]
    fold3_tags_train = fold3_tags_train[:,:,0]
    fold4_tags_train = fold4_tags_train[:,:,0]

    Y_fold1 = Y_fold1[:,:,0]
    Y_fold2 = Y_fold2[:,:,0]
    Y_fold3 = Y_fold3[:,:,0]
    Y_fold4 = Y_fold4[:,:,0]

    levels_tags_fold1 = levels_tags_fold1[:,:,0]
    levels_tags_fold2 = levels_tags_fold2[:,:,0]
    levels_tags_fold3 = levels_tags_fold3[:,:,0]
    levels_tags_fold4 = levels_tags_fold4[:,:,0]

    # Fill the dictionary fold-wise
    # 1-fold
    folds_dict['1-fold']['X_train'] = fold1_X_train.astype('float64')
    folds_dict['1-fold']['Y_train'] = fold1_Y_train.astype('float64')
    folds_dict['1-fold']['X_test'] = X_fold4.astype('float64')
    folds_dict['1-fold']['Y_test'] = Y_fold4.astype('float64')
    folds_dict['1-fold']['train_tags'] = fold1_tags_train
    folds_dict['1-fold']['test_tags'] = levels_tags_fold4

    # 2-fold
    folds_dict['2-fold']['X_train'] = fold2_X_train.astype('float64')
    folds_dict['2-fold']['Y_train'] = fold2_Y_train.astype('float64')
    folds_dict['2-fold']['X_test'] = X_fold3.astype('float64')
    folds_dict['2-fold']['Y_test'] = Y_fold3.astype('float64')
    folds_dict['2-fold']['train_tags'] = fold2_tags_train
    folds_dict['2-fold']['test_tags'] = levels_tags_fold3

    # 3-fold
    folds_dict['3-fold']['X_train'] = fold3_X_train.astype('float64')
    folds_dict['3-fold']['Y_train'] = fold3_Y_train.astype('float64')
    folds_dict['3-fold']['X_test'] = X_fold2.astype('float64')
    folds_dict['3-fold']['Y_test'] = Y_fold2.astype('float64')
    folds_dict['3-fold']['train_tags'] = fold3_tags_train
    folds_dict['3-fold']['test_tags'] = levels_tags_fold2

    # 4-fold
    folds_dict['4-fold']['X_train'] = fold4_X_train.astype('float64')
    folds_dict['4-fold']['Y_train'] = fold4_Y_train.astype('float64')
    folds_dict['4-fold']['X_test'] = X_fold1.astype('float64')
    folds_dict['4-fold']['Y_test'] = Y_fold1.astype('float64')
    folds_dict['4-fold']['train_tags'] = fold4_tags_train
    folds_dict['4-fold']['test_tags'] = levels_tags_fold1

    return folds_dict

