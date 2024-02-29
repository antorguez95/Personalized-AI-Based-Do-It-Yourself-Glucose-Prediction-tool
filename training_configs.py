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

from sensor_params import * # All sensors parameters

# Dictionary of all the possible training configurations to be used in the loop
big_test = {'sensor' : [sensor_Mikael],
                'N' : [48, 96, 144], 
                'step' : [1], 
                'PH' : [5, 30, 60], 
                'single_multi_step' : ['single', 'multi'],
                'partition' : ['june-21', 'month-wise-4-folds'],
                'normalization' : [None, 'min-max'],
                'under_over_sampling' : ['under', None], 
                'model' : ['1D-UNET', '1D-UNET-non-compres', 'DIL-1D-UNET', 'LSTM', '1D-UNET-LSTM', 'StackedLSTM'],
                'loss_function' : ['root_mean_squared_error', 'ISO_loss'], 
                }

loss_functions_comparison = {'sensor' : [sensor_Mikael],
                'N' : [96], 
                'step' : [1], 
                'PH' : [60, 30, 5], 
                'single_multi_step' : ['single'],
                'partition' : ['june-21'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['1D-UNET-non-compres', 'LSTM', 'StackedLSTM'],
                'loss_function' : ['ISO_loss', 'root_mean_squared_error'], 
                }

stacked_LSTM_multi = {'sensor' : [sensor_Mikael],
                'N' : [96], 
                'step' : [1], 
                'PH' : [60, 30], 
                'single_multi_step' : ['multi'],
                'partition' : ['june-21'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['StackedLSTM'],
                'loss_function' : ['ISO_loss', 'root_mean_squared_error'], 
                }

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

N_patients_N_models_DL = {'sensor' : [libreview_sensors],
                'N' : [96], 
                'step' : [1], 
                'PH' : [60],# [30, 60], 
                'single_multi_step' : ['multi'],
                'partition' : ['month-wise-4-folds'],
                'normalization' : ['min-max'],
                'under_over_sampling' : [None], 
                'model' : ['naive', 'LSTM', 'StackedLSTM', 'DIL-1D-UNET'],#['naive', 'LSTM', 'StackedLSTM', 'DIL-1D-UNET'],# '1D-UNET', '1D-UNET-non-compres', 'DIL-1D-UNET', 'LSTM'],
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


