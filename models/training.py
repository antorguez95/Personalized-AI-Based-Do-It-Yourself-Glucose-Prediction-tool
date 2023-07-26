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
import time 
import pickle
import matplotlib.pyplot as plt 

# Add the parent directory to the path 
import sys
sys.path.append('..')
from evaluation.multi_step.evaluation import *

from typing import Tuple, Dict

from sensor_params import *
from arch_params import * 

import pandas as pd
import numpy as np


def iso_percentage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """This computes the positive percentage of the points within the acceptable
    range according to the ISO___ [1]. It expects the arguments to be generated
    during the Keras fit/evaluate runtime, so first it computes the obtained
    glucose values and then computes both percentage calling ____ for the entire batch.  

    Args:
    -----
        y_true: The true glucose values.
        y_pred: The predicted glucose values.
    
    Returns:
    --------
        iso_perc: The percentage of points within the acceptable range.
        parker_perc: The percentage of points within the acceptable range in the Parker Grid.

    References:
    -----------
        [1] ISO __________
    """
    # If the batch size is None, return 0 for both metrics
    if y_true.shape[0] == None:
        return 0, 0
    
    # for i in tqdm(range(y_true.shape[0]), desc="Computing PPV and Sens"):

    # Compute the ISO and Parker percentages
    iso_percentage, _ , _= bgISOAcceptableZone(y_true, y_pred)
    parker_percentage, _, _= parkes_EGA_chart(y_true, y_pred, "_")
    
    return iso_percentage, parker_percentage

class CustomMetrics(tf.keras.callbacks.Callback):
    """Custom callback to calculate the positive predicted value and sensitivity
    metrics in the validation data after each epoch.
    """

    def __init__(self, data, prefix=''):
        super(CustomMetrics, self).__init__()
        self.data = data
        self.prefix = prefix
        self.iso_perc = []
        self.parker_perc = []

    def on_epoch_end(self, epoch, logs=None):

        x, y = self.data
        _iso_percentage, _parker_percentage = iso_percentage_metrics(y, self.model.predict(x))
        
        self.iso_perc.append(_iso_percentage)
        self.parker_perc.append(_parker_percentage)
        print('- {}ISO %: {} - {}Parker %: {}'.format(self.prefix, _iso_percentage, self.prefix, _parker_percentage), end=' ')

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

def train_model(model : tf.keras.Model,
                X: np.array, 
                Y: np.array,
                N: int, 
                stride: int,
                kernel_size: int, 
                predicted_points: int,
                sensor : Dict, 
                epochs: int, 
                batch_size: int, 
                lr: float,
                fold : int,
                verbose: int = 1) -> None: 
    """Train a previously loaded Deep Learning model parameters using 
    the given data, and some model hyperparameters. 

    Args:
    -----
        model (tf.keras.Model): The model to train.
        X (np.array): The input features (size = N).
        Y (np.array): The output sequence (size = predicted_points).
        N (int): Input features length.
        stride (int): Stride of the windowing.
        kernel_size (int): Kernel size of the convolutional layers.
        predicted_points (int): Number of points to predict, i.e., the output sequence length.
        epochs (int): Number of epochs.
        sensor (Dict): Dictionary with the sensor name and its parameters.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        fold (int): When training with cross-validation, the fold to train and save the model with its name. 
        verbose (int): Verbosity level. Defaults to 1.

    Returns:
    --------
        None
    """
    
    # Model compile 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.MeanSquaredError(), 
                metrics=[tf.keras.metrics.RootMeanSquaredError()]) # Maybe we can explore more
    
    # Modular path of the model. Current directory assumed to be the parent directory 
    dir = os.getcwd()
    
    # Create directory if it has not been previously created
    if 'training' not in dir:  
        # Create directory to store the training parameters and results
        training_path = r'\training'
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
        f.write('tau = {}\n'.format(tau))
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

    # Create a callback to evaluate the ISO percentages after each epoch   
    custom_metrics_train = CustomMetrics([X, Y])
    # callbacks.append(custom_metrics_train)

    # Callback to implement early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', 
                                                min_delta =0.5,
                                                patience=1, # epochs to wait until stop
                                                mode = "min",
                                                restore_best_weights=True,
                                                )
    # Model training
    history = model.fit(
            x=X,
            y=Y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks= callback,
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

    # Save custom metrics too
    with open(str(fold)+'_custom_metrics_train.pkl', 'wb') as f:
        pickle.dump({'ISO%': custom_metrics_train.iso_perc, 'Parker%': custom_metrics_train.parker_perc}, f)

    # Plot training and validation loss
    fig, axs = plt.subplots(4,1, sharex=True, figsize=(10,5))
    axs[0].plot(history.history['loss'], label='Training')
    axs[0].set_ylabel('Loss, RMSE')

    axs[0].legend()

    axs[1].plot(history.history['root_mean_squared_error'], label='Training')
    axs[1].set_ylabel('RMSE')

    # Plot custom metrics in the two last axes
    axs[2].plot(custom_metrics_train.iso_perc, label='Training')
    axs[2].set_ylabel('ISO %')

    axs[3].plot(custom_metrics_train.parker_perc, label='Training')
    axs[3].set_ylabel('Parker %')

    axs[-1].set_xlabel('Epoch')

    # Reduce the space between the subplots
    fig.subplots_adjust(hspace=0)

    plt.suptitle(str(fold)+' Training evolution: N = {}, tau = {}, PH = {} min.' .format(N, tau, predicted_points*5))

    # Save the figure
    plt.savefig(dir+'/'+str(fold)+'_training_evolution.png')

    # Model name (depends on the fold to train)
    model_name = '/'+str(fold) +'.h5'
    
    # Save the model
    model.save(dir+model_name)

    if verbose >= 1:
        print('\tEnd of the training. Model saved in {}'.format(dir))
