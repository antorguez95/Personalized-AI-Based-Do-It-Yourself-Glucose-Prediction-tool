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

import numpy as np
from sensor_params import *
from typing import *
import copy

def naive_model(X : np.array, input_feature : int, pred_steps : int) -> np.array:
    """Returns a naive model that predicts a sequence that it 
    is just the input array shifted a number of positions equal to
    Prediction Horizon (PH) / sampling period of the sensor.
    This is just for comparison purposes

    Args:
    -----
        X (np.array): Input tensor.
        input_feature (int): Number of features in the input tensor.
        pred_steps(int): Number of points to predict, that corresponds to: PH/sampling frequency of
        the sensor.

    Returns:
    --------
        naive_prediction (np.array): Naive prediction.
    """

    # Take only the last PH/sensor["SAMPLE_PERIOD"] samples of the input array (Equivalent to make a prediction that is only a shift of the input array)
    if input_feature ==  1:
        naive_prediction = copy.deepcopy(X)
        naive_prediction = naive_prediction[:,-(round(pred_steps)):]
    
    elif input_feature > 1:
        naive_prediction = copy.deepcopy(X[:,:,0])
        naive_prediction = naive_prediction[:,-(round(pred_steps)):]

    return naive_prediction