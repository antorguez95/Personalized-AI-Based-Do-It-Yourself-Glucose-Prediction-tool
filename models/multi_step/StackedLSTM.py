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

from tensorflow.keras import layers, Input, Model
from typing import Dict

from arch_params import *

# Returns a LSTM-model instance 
def get_model(sensor : Dict, N: int = CGM_INPUT_POINTS, input_features: int = NUMBER_OF_INPUT_SIGNALS,
            PH : int = 1) -> Model:
    """Returns a Stacked LSTM [1] for CGM multistep forecasting whose number of 
    memory units depends on the lenght on the input tensor.

    Args:
    -----
        sensor (Dict) : Dictionary with the sensor's information, such as the sampling frequency.
        N (int): Number of samples in the input tensor. Must be multiple of 2. Default: CGM_INPUT_POINTS.
        input_features (int): Number of features in the input tensor. Default: NUMBER_OF_INPUT_SIGNALS.
        PH (int): Prediction Horizon to predict. Length of the predicted sequence lenght = PH/sampling frequency of
        the sensor. Default: 5.
        sensor (Dict) : Dictionary with the sensor's information. Default: CGM_SENSOR. ##### TO BE INCLUDED
    
    Returns:
    --------
        model (Model): Stacked LSTM-model instance.
    
    References:
    -----------
        [1] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. 
        Neural Comput. 9, 8 (November 15, 1997), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735.
    """
    # Input tensor
    input = Input(shape=(N, input_features)) 

    # LSTM 
    x = layers.LSTM(round(N/2), input_shape = (N, input_features), return_sequences = True)(input)
    x = layers.LSTM(round(N/4), input_shape = (N, input_features), return_sequences = True)(x)
    x = layers.LSTM(round(N/8), input_shape = (N, input_features))(x)

    # Dense layer that outputs the predicted points
    output = layers.Dense(PH/sensor["SAMPLE_PERIOD"])(x) # PH/SENSOR_SAMPLING_FREQUENCY

    # Define the model
    model = Model(input, output)

    return model