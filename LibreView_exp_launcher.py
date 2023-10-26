# Copyright (C) 2023 Antonio J. Rodriguez-Almeida
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

# Usage:
# launch_experiment.py [-h]
#                   (--training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE)
#                   [--model_hyperparameters KERNEL_SIZE STRIDE] 
#                   [--dataset_folder DATASET_PATH]
#                   {big_test,loss_functions_comparison,test, stacked_LSTM_multi}
#
# 
# Launch multiple experiments to train and evaluate multiple CGM forecast DL models passing their configuration through the command line.

# positional arguments:
#   {big_test,loss_functions_comparison,test, stacked_LSTM_multi}
#                         Dictionaries with the configuration. (Modify 'training_configs.py' as you wish)

# options:
#   -h, --help            show this help message and exit
#   --training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE
#                         Use custom training hyperparameters
#   --model_hyperparameters KERNEL_SIZE STRIDE
#                         Use custom model hyperparameters
#   --dataset_folder DATASET_PATH
#                         Specify the dataset directory

import argparse
from main_libreview import launch_LibreView_experiments
from training_configs import *

epochs_default = 10
batch_size_default = 2
learning_rate_default = 0.0001

kernel_size_default = 3
tau_default = 1

DATASET_PATH = r"C:\Users\aralmeida\Downloads\LibreViewRawData"

# Parse command line arguments
parser = argparse.ArgumentParser(prog='Train and Evaluate DL-based CGM Forecast models',
                                 description="Launch multiple experiments to train and evaluate multiple CGM forecast DL models passing their configuration through the command line.")

# Add the training configuration as a mandatory argument 
parser.add_argument("training_config", help='Dictionaries with the configuration. (Modify \'training_configs.py\' as you wish)',
                    choices=['big_test', 'loss_functions_comparison', 'test', 'stacked_LSTM_multi', 'LibreView-exp-1'])

# Add the training hyperparameters as optional arguments
parser.add_argument("--training_hyperparameters", nargs=3, help='Use custom training hyperparameters', metavar=('EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE'), default=[epochs_default, batch_size_default, learning_rate_default])

# Add the model hyperparameters as optional arguments
parser.add_argument("--model_hyperparameters", nargs=2, help='Use custom model hyperparameters', metavar=('KERNEL_SIZE', 'STRIDE'), default=[kernel_size_default, tau_default], type=int)

# Add the dataset directory as optional arguments
parser.add_argument("--dataset_folder", nargs=1, help='Specify the dataset directory', metavar=('DATASET_PATH'), default=[DATASET_PATH], type=str)

args = parser.parse_args()

# Catch the dictionary with the desired experiment
if args.training_config == 'big_test':
    training_config = big_test
elif args.training_config == 'loss_functions_comparison':
    training_config = loss_functions_comparison
elif args.training_config == 'stacked_LSTM_multi':
    training_config = stacked_LSTM_multi
elif args.training_config == 'test':
    training_config = testing
elif args.training_config == 'LibreView-exp-1':
    training_config = N_patients_N_models_DL

else:
    raise ValueError('The training configuration is not valid')

# Catch the training hyperparameters
epochs = int(args.training_hyperparameters[0])
batch_size = int(args.training_hyperparameters[1])
learning_rate = float(args.training_hyperparameters[2])

# Catch the model hyperparameters 
kernel_size = int(args.model_hyperparameters[0])    
tau = int(args.model_hyperparameters[1])

# Catch the dataset directory
DATASET_PATH = args.dataset_folder[0]

# Call the experiments launcher function
launch_LibreView_experiments(training_config, kernel_size, tau, learning_rate, batch_size, epochs)