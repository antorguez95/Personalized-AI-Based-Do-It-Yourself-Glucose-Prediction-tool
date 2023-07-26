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
import tensorflow as tf
from typing import Tuple

from arch_params import *





# Returns a CNN-model instance 
def get_model(N: int = CGM_INPUT_POINTS, input_features: int = NUMBER_OF_INPUT_SIGNALS,
            predicted_points : int = 1) -> Model:
    """Returns the model described in [1].

    Args:
    -----
        N (int): Number of samples in the input tensor. Must be multiple of 2. Default: CGM_INPUT_POINTS.
        input_features (int): Number of features in the input tensor. Default: NUMBER_OF_INPUT_SIGNALS.
        tau (int): Stride of the convolutional layers. Default: 1, as [1]
        kernel_size (int): Kernel size of the convolutional layers. Default: 3, as [1]
        output_points (int): Number of predictied points (time dimension) Default: 1.
    
    Returns:
    --------
        model (Model): CNN-model instance.
    
    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """
    # Input tensor
    input = Input(shape=(N, input_features)) 

    # LSTM 
    x = layers.LSTM(round(N/2), input_shape = (N, input_features), return_sequences = True)(input)
    x = layers.LSTM(round(N/4), input_shape = (N, input_features))(x)

    # Dense layer that outputs the predicted points
    output = layers.Dense(predicted_points)(x)

    # Define the model
    model = Model(input, output)

    return model