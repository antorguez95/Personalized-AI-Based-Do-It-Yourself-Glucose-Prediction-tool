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

from arch_params import *

# Parameters (dependent on the sensors) for the input data that will fix the architecture of the tested ML/DL models

# Mikael sensor
sensor_Mikael = {
    "NAME" : "Mikael sensor",
    "INPUT_SIZE" : CGM_INPUT_POINTS,
    "INPUT_FEATURES" : 1,
    "STRIDE" : 1,
    "SAMPLE_PERIOD" : 5, #minutes between consecutive readings
    }

# Dummy sensor 
DUMMY_SAMPLE_PERIOD = 1 #minutes per time

sensor_Dummy = {
    "NAME" : "TEST",
    "INPUT_SIZE" : CGM_INPUT_POINTS,
    "INPUT_FEATURES" : 1,
    "STRIDE" : 1,
    "SAMPLE_PERIOD" : DUMMY_SAMPLE_PERIOD, #times per minute
    "PREDICTED_POINTS" : PREDICTION_TIME/DUMMY_SAMPLE_PERIOD # means 2 * SAMPLE_PERIOD MINUTES
    }

# RMSE SE PUEDE QUITAR LA RAIZ 
# minimo del error cuadratico 


