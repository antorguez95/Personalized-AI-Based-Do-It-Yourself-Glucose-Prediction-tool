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

# sensor_params.py
# This module contains as many dictionaries as type of sensors included in the 
# study. In this case, only libreview_sensors was included because it 
# represents all the included sensors (only CGM entry and 15 minutes of sampling period). 
# 
# Introduce here your sensor to perform this experiment. See README.md to learn the
# full process. An example is included in your_new_sensor variable. 

libreview_sensors = {
    "NAME" : "FreeStyle Libre X",
    "CGM" : True,
    "INSULIN" : False, 
    "SAMPLE_PERIOD" : 15, 
}

# Example of a new sensor. 
your_new_sensor = {
    "NAME" : "FreeStyle Libre 3",
    "CGM" : True,
    "INSULIN" : True, # If you want to include insulin too, you have to change the code accordingly. 
    "SAMPLE_PERIOD" : 5, #minutes between consecutive readings
    }



