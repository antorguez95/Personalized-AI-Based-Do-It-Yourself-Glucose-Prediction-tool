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

from tensorflow.keras import layers, Input, Model
from typing import Dict

from arch_params import *

# Returns a LSTM-model instance 
def get_model(N: int, input_features: int = 1) -> Model:
    """Returns a Stacked LSTM [1] for CGM single-step forecasting whose number of 
    memory units depends on the lenght on the input tensor.

    Args:
    -----
        N (int): Number of samples in the input tensor. Must be multiple of 2. 
        input_features (int): Number of features in the input tensor. Default: 1.
    
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
    output = layers.Dense(1)(x) # PH/SENSOR_SAMPLING_FREQUENCY

    # Define the model
    model = Model(input, output)

    return model