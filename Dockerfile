# Python version 
FROM python:3.10.13
# FROM scratch

# Maintainer 
MAINTAINER Antonio Rodriguez <aralmeida@iuma.ulpgc.com>

# Work directory 
# WORKDIR ${.\testing_with_mirek}
WORKDIR /code

# Copy the file containing the environment description 
# COPY environment.yml ./
COPY ./requirements.txt /code/requirements.txt

# Copy libraries 
COPY ./app /code/src/app
COPY ./evaluation/ /code/src/evaluation
COPY ./models /code/src/models
COPY ./utils.py /code/src/utils.py
COPY ./sensor_params.py /code/src/sensor_params.py
COPY ./training_configs.py /code/src/training_configs.py
COPY ./your_AI_DIY_parameters.py /code/src/your_AI_DIY_parameters.py

# From conda create the environment and the required packages 
# RUN pip
# RUN conda env create -n diy_app_env -f environment.yml
# To obtain the requirements.txt file : pip freeze > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

# Add the test file (only one)
COPY ./DIY_top_module.py /code/src/DIY_top_module.py

CMD ["python", "src/DIY_top_module.py"]


