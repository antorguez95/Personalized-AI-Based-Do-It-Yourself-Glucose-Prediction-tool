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


# Returns a model instance 
def get_model(N: int = CGM_INPUT_POINTS, input_features: int = NUMBER_OF_INPUT_SIGNALS) -> Model:
    """Returns a simple Dense Layer to serve as baseline to Deep Learning models
    single step forecasting evaluation. The Prediction Horizon of the model 
    is defined by the previously generated training dataset since it does not influence
    the model's architecture.

    Args:
    -----
        N (int): Number of samples in the input tensor. Must be multiple of 2. Default: CGM_INPUT_POINTS.
        input_features (int): Number of features in the input tensor. Default: NUMBER_OF_INPUT_SIGNALS.
 
    Returns:
    --------
        model (Model): Dense layers based model instance.
    
    """
    # Input tensor
    input = Input(shape=(N, input_features)) 

    # Flatten x
    x = layers.Flatten()(x)

    # Couple of dense layers to reduce the dimensionality
    x = layers.Dense(32)(x)
    x = layers.Dense(8)(x)
    x = layers.Dense(2)(x)

    # Once flattened, add a dense layer to predict the output
    output = layers.Dense(1)(x)

    # Define the model
    model = Model(input, output)


    return model