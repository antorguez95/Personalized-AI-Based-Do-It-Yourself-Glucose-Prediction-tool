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

# Returns a CNN-model instance 
def get_model(N: int, input_features: int = 1) -> Model:
    """RReturns a LSTM [1] for CGM single step forecasting whose number of memory units 
    depends on the lenght of the input features (N).The Prediction Horizon of the model 
    is defined by the previously generated training dataset since it does not influence
    the model's architecture.

    Args:
    -----
        N (int): Number of samples in the input tensor. Must be multiple of 2.
        input_features (int): Number of features in the input tensor.

    
    Returns:
    --------
        model (Model): LSTM-model instance.
    
    References:
    -----------
        [1] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. 
        Neural Comput. 9, 8 (November 15, 1997), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735.
    """
    # Input tensor
    input = Input(shape=(N, input_features)) 

    # LSTM 
    x = layers.LSTM(round(N/4), input_shape = (N,1))(input)

    # Dense layer that outputs the predicted points
    output = layers.Dense(1)(x)

    # Define the model
    model = Model(input, output)


    return model