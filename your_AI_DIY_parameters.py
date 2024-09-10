from sensor_params import *

first_DIY_version = {'N' : 96, # Input window length
                'step' : 1, # Step to sweep the CGM data to generate the dataset
                'PH' : 30, # Prediction Horizon 
                'single_multi_step' : 'multi', # 'multi' if more than one step ahead is performed. 'single' for one step. 
                'partition' : 'month-wise-4-folds', # Cross-Validation partition (to add more, the correspondent functions must be implemented)
                'input_features' : 2, # Currently 1 means only CGM, and 2 means CGM and its derivative
                'normalization' : 'min-max', # Normalization (to add more, the correspondent functions must be implemented)
                'under_over_sampling' : None, # Currently only undersampling was implemented, without positive results
                'model' : 'LSTM', # Currently: 'DIL-1D-UNET' or 'StackedLSTM' supported. Add whatever you want in 'models' folder and add it in the main_libreview.py
                'loss_function' : 'ISO_loss' # Loss function to generate your model. 'root_mean_squared_error' (actually MSE) is also supported 
                }

# PLEASE, CHECK THE IMPLICATIONS OF THESE CHANGES ON THE DIY_top_module.py code, since not all of this is currently parametrized.