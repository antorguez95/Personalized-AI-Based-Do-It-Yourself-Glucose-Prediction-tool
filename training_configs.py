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

# training_configs.py
# This module contains as many dictionaries as training configurations
# you are willing to test, providing high flexibility and enabling 
# comparison between architectures, PHs, input winwos lengths, etc. 
# This is the description of each field: 
#       - sensor: dictionary with the sensor parameters (see sensor_params.py).
#       - N: input window length (i.e., number of samples). 
#       - step: step between consecutive samples in the dataset generation. 
#       - PH: prediction horizon.
#       - single_multi_step: single or multi-step prediction.
#       - partition: partition strategy.
#       - normalization: normalization strategy.
#       - under_over_sampling: under or over sampling strategy.
#       - model: model to be used.
#       - loss_function: loss function to be used.
#
# In case you want to reproduce and/or compare your results with us, the 
# N_patients_N_models_DL is the one that represents the experimentation 
# included in our paper. 
# 
# Introduce here your dicionary with your desired configuratoin. See README.md
# to learn the full process. An example is included in your_new_sensor variable. 

from sensor_params import * # All sensors parameters

# Dictionary of all the possible training configurations to be used in the loop
first_approach = {'sensor' : [libreview_sensors],
                'N' : [96], 
                'step' : [1], 
                'PH' : [15, 30, 60], 
                'single_multi_step' : ['multi'],
                'partition' : ['month-wise-4-folds'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['naive', 'StackedLSTM'],
                'loss_function' : ['ISO_loss', 'root_mean_squared_error'], 
                }

testing = {'sensor' : [libreview_sensors],
                'N' : [96, 144], 
                'step' : [1], 
                'PH' : [30, 60], 
                'single_multi_step' : ['multi'],
                'partition' : ['month-wise-4-folds'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['LSTM'],
                'loss_function' : ['root_mean_squared_error'], 
                }

###### THIS IS THE CONFIGURATION OF THE EXPERIMENT INCLUDED IN THE PAPER!!!!! #######
N_patients_N_models_DL = {'sensor' : [libreview_sensors],
                'N' : [96], 
                'step' : [1], 
                'PH' : [30, 60], 
                'single_multi_step' : ['multi'],
                'partition' : ['month-wise-4-folds'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['naive', 'LSTM', 'DIL-1D-UNET', 'StackedLSTM'], 
                'loss_function' : ['ISO_loss', 'root_mean_squared_error'], 
                }

only_naive = {'sensor' : [libreview_sensors],
                'N' : [96], 
                'step' : [1], 
                'PH' : [30, 60], 
                'single_multi_step' : ['multi'],
                'partition' : ['month-wise-4-folds'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['naive'],
                'loss_function' : ['root_mean_squared_error'], 
                } 