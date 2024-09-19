# Python version 
FROM python:3.10.13
# FROM scratch

# Maintainer 
MAINTAINER Antonio Rodriguez <aralmeida@iuma.ulpgc.com>

# Work directory 
# WORKDIR ${.\testing_with_mirek}
WORKDIR /CGM_forecasting

# Copy the file containing the environment description 
COPY ./requirements.txt /CGM_forecasting/requirements.txt


# Copy AI-based DIY module libraries  
COPY ./app /CGM_forecasting/code/src/app
COPY ./evaluation/ /CGM_forecasting/code/src/evaluation
COPY ./models /CGM_forecasting/code/src/models
COPY ./utils.py /CGM_forecasting/code/src/utils.py
COPY ./sensor_params.py /CGM_forecasting/code/src/sensor_params.py
COPY ./training_configs.py /CGM_forecasting/code/src/training_configs.py
COPY ./your_AI_DIY_parameters.py /CGM_forecasting/code/src/your_AI_DIY_parameters.py
COPY ./drop_your_data_here_and_see_your_pred /CGM_forecasting/drop_your_data_here_and_see_your_pred
#COPY ./here_is_your_prediction /CGM_forecasting/here_is_your_prediction


# From conda create the environment and the required packages 
# To obtain the requirements.txt file : pip freeze > requirements.txt (some things were commented)
RUN pip install --no-cache-dir -r requirements.txt 

# Add the app top module file 
COPY ./DIY_top_module.py /CGM_forecasting/code/src/DIY_top_module.py

CMD ["python", "code/src/DIY_top_module.py"]


