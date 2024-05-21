# Copyright (C) 2024 Antonio J. Rodriguez-Almeida
# 
# This file is part of _______.
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

def DIY_glucose_prediction_top_module(user_file : str) -> None: 
    
    """
    This function encapsulates the Do It Yourself (DIY) module for glucose prediction.
    After experimentation (see main_libreview.py), this function can be considered 
    the DIY module itself ready to be used. The input of the module is a .cvs (or .json) file.
    The output of this module is: 
        a) If it is the user's first use, it runs the normal training of the DL model following a
        4-folds month-wise CV approach. The model that presents better performance is saved as a .h5 file.
        After this, T1D-related personal data analysis an visualization is provided, and
        an 1-hour prediction is performed and depicted using the last 24 hours of the user's data (96 data points
        at 15-minute intervals for the LibreView data used to develop this framework). 
        ***** THE CHOICE OF THE BEST MODEL MIGHT CHANGE IN SUBSEQUENT VERSIONS OF THIS FRAMEWORK *****
        b) If the user has already a DL model, (i.e., step a) has been done once) the function loads the model,
        performs the same analysis with the new updated user's data, and performs a new 1-hour prediction taking 
        the last 24 hours of the user's data.
    
    Args: 
    ----
        user_file : user's file. This file should be a .csv or .json file with the user's data. ***More formats to come
    
    Returns:
    -------
        results : a dictionary to visualize the results even no new data is provided by the user, but whenever the user wishes so. 
    
    """


    # Case 1: User's first use and enough data to train the model (set threshold)

    # Case 2: User's first use and not enough data to train the model 

    # Case 3: User's second (or beyond) use and enough data to call the model (set now to 96 samples/24 hours)

    # Case 4: User's second (or beyond) use and not enough data to call the model (less than one day recorded with no interruptions)

