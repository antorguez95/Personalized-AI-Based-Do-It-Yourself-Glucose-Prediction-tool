# An AI-based "Do-it-Yourself" tool for personalized glucose prediction.
Hi! If you got here, it is very likely that you are interested in blood glucose monitoring, AI, or both! If it is the first case and you are willing to use this tool to predict your interstitial glucose level with your personalized AI model, [go directly to the *Use me!* section](#Use-me-!). If, on the contrary, you are more interested in the development of this tool, the code, or you want to test your own models in this framework, you better go to [the developers and researchers section](#For-developers-and-researchers). 

This module has been developed with data from people with **type 1 diabetes**, but of course it is suitable for anyone who has a diabetes-related condition that implies having a glucose sensor attached to your body. 

If you are interested to read about the scientific basis of this work, please check our paper [*"An AI-Based “Do-It-Yourself” Module for Interstitial Glucose Forecasting for People with Type 1 Diabetes"*](https://ieeexplore.ieee.org/document/9851514) published on [npj Digital Medicine](https://www.nature.com/npjdigitalmed/) scientific journal. Besides, if this work somehow helped you with your research work or with a personal project, please, cite out paper.  

## Use me!  
### First time?
First time is always special, and this is not an exception. The first time you use this module, you will get your personalized AI-glucose predictor. Once you have generated your model, you just have to follow the instructions on [Use me (again)! section](#Use-me-(again)!). Currently, this tool only supports the following sensors: 

#### Included sensors
| Data source  | Sensors | Input data | 
| ------------- | ------------- | ------------- |
| LibreView app | FreeStyle Libre | Glucose | 
| LibreView app | FreeStyle LibreLink | Glucose | 
| LibreView app | LibreLink | Glucose | 

At this point, we assume that you have access to your real CGM data. If not, type in the terminal `python get_your_toy_cgm_file.py` to generate a LibreView-like CGM `.csv` file and play with this module! Same if your sensor is not included in the previous table (we're sorry, we are working on it!). Of course, to run this script, you need to have Python installed on your PC!

#### What do you need?
We have designed this module to be the least overwhelming possible as to you. So, to install and execute this module you just need a PC, an open terminal, an the [Docker Desktop installed in your PC](https://docs.docker.com/desktop/). Once you have downloaded it, you can proceed with the next steps. If everything goes as it should and as far as **you don't replace your CGM sensor**, you only will have to perform these steps **just ONCE**: 

1) Open your Docker Desktop (if you are not in an admin account, right click and click on "Run as administrator").
2) Open a terminal. For example, typing "cmd" from the "Search" bar in Windows.
3) In the terminal, go to the directory where you want to install and save this module together with your personalized AI-model. Let's asume the directory `C:\Users\aralmeida`:

```
cd C:\Users\aralmeida
```
   
5) Now you are in your directory. Let's build the Docker image [(go here if you are interested on this)](https://docs.docker.com/get-started/) to run your model later! Copy and paste the following line (the final dot is not a mistake!). After a few minutes, all you need to have your personalized AI model has been installed. 

```
docker build -t diy_cgm_image .
```

7) **IMPORTANT**: Create a folder named `/drop_your_data_here_and_see_your_pred`. Place it whenever you want in your PC. This folder is the one that will always be accesed by this app to execute the model and perform your prediction. **DO NOT MOVE, REMOVE, OR CHANGE THE NAME of the folder** If you do so, you will have to recreate all this steps with the new directory.
8)  Now, drop the file with your CGM data (usually ended with `.csv`) in the recently created `/drop_your_data_here_and_see_your_pred` folder.
9)  We are ready to execute this app for the first time. Let's assume that the path were you placed your `/drop_your_data_here_and_see_your_pred` folder is: `C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred`. As you can see, I created my folder just in the common `Downland`folder in Windows. Now, you have to type (or copy/paste) this in your terminal **with your own directory, of course!**:

```
docker run -ti --mount type=bind,source=/C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred,target=/CGM_forecasting/drop_your_data_here_and_see_your_pred diy_cgm_image
```

**WARNING**: if you, let's say, change the folder (for example) from `C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred` to `C/Users/aralmeida/Desktop/drop_your_data_here_and_see_your_pred`, you should replace `source=/C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred`by `source=/C/Users/aralmeida/Desktop/drop_your_data_here_and_see_your_pred`. Just that.  

8) If everything went good, now you should see some interactive stuff on your terminal. You will be asked about the model of your CGM sensor, the glucose concentration units you are used to, or if you want to know a little bit more about the relation of your CGM data and the AI. At this point, two different things could happen: 
   -  **THE IDEAL CASE**: the year that will be used to generate and train your AI model do not contain too may interruptions an this app is able to generate a reliable AI-based predictor.
   -  **BAD NEWS**: the year extracted from your uploaded CGM data contains too many interruptions to reliably generate an AI model. This is maybe due to sensor disonnections, or misfunctions... We are sorry! The good news is that you can always go to the first step when you have more data available!

9) Let's assume the best scenario. You were able to generate your AI model (after a few hours...). If you gou to your `/drop_your_data_here_and_see_your_pred` you will see a few things. You just have to focus on two.
   - `your_AI_based_CGM_predictor.h5`. This is **your personalized AI model for your glucose level predictions!** Isn't it great?? Please, do not delete this file. If you do so, you will have to repeat all this process.
   - `your_last_prediction.png`. This is an example prediction. Of course, this prediction is not useful, since it goes after the model generation, that took a couple of hours. As you can see, the model takes the data from your last **24 hours**, and performs a predcition of your next 30'. The red shadow shows the error of each instant, that has been calculated from your data. From now on, everytime you execute this app, this picture will be overwritten with your most recent prediction, so this is what you will see! Unfortunately, if your last day contains CGM reading interruptions, the model will not be able to provide you with a prediction.

Until now, everytime you want to use this app, you have to follow the simple steps described in [Use me (again)! section](#Use-me-(again)!). 

### Use me (again)!

Hi again! If you are here, it means you have `your_AI_based_CGM_predictor.h5` file in your `/drop_your_data_here_and_see_your_pred` folder, and you want to get now a personalized prediction of your next 30 minutes. 
   1) Download your data in a `.csv` file, containing at least your last 24 hours of CGM readings.
   2) Drop it in the `/drop_your_data_here_and_see_your_pred` folder.
   3) Open a terminal in your PC and execute the following lines (remember to replace 'APP_DIRECTORY' (same as step 3 of [Use me! section](#Use-me!)) and 'YOUR_DATA_DIRECTORY/drop_your_data_here_and_see_your_pred' by the directory where you save your CGM file:
   ```
   cd APP_DIRECTORY
   ```
   
   ```
   docker run -ti --mount type=bind,source=/YOUR_DATA_DIRECTORY/drop_your_data_here_and_see_your_pred,target=/CGM_forecasting/drop_your_data_here_and_see_your_pred diy_cgm_image
   ```
   4) Now, the terminal will promtp you with some messages. If you had CGM readings interruptions in your last 24 hours, we cannot provide a reliable prediction. We are sorry! If your data was OK, please check the `your_last_prediction.png` picture with your updated prediction. And, please, never take this as an absolute truth! This is jsust an AI-based prediction!

From now on, there are the steps you have to follow any time you want to have a personalized glucose prediction based on your own AI model! Hope this help you with your diabetes! 

## For developers and researchers
Hi again! If you are here, it is assumed that you have basic knowledge of Python programming and AI, so let's get into it!

The framework to design this _"Do-It-Yourself"_ AI based module is based on Python 3.10 and TensorFlow 2.6.0. Typical libraries for Machine Learning development, such as scikit-learn, or matplotlib for visualization. All the requirements are icluded on `requirement.txt` file. Notice that if you don't use GPU, cuda libraries won't be used. So, if you want to explore the code and/or play with it, introduce your own models, change the data generation parameters, etc., your are in the right place! First of all, create a conda environment. 

**IMPORTANT!!** Go to the [TensorFlow website](https://www.tensorflow.org/install/source_windows?hl=es-419#gpu) to check the compatibilities between Python, CUDA, cuDNN and Tensorflow versions. 

```
conda create -n DIY_for_CGM_pred python=3.10
```
Then, install all the required libraries. 
```
pip install --no-cache-dir -r requirements.txt 
```
Now, you before running any experiments, you would need some CGM data. If you don't have any '.csv' file directly downloaded from the LibreView app, which could happen, execute the following script: 

```
python get_your_toy_cgm_file.py
```
In your current directory, you should see a file with a name following the format that imitates LibreView-generated files: `IDXXX_SXXX_RXXX_glucose_DD-MM-YYY.csv`. Inside, there are fake CGM readings that do not follow any CGM patterns, just random numbers between the physiological glucose level limits. This data is in 'mg/dL'. 

Now, you are able to run experiments. It is desirable for you to get access to real CGM data to get real conclusion from this. But, in the meantime, this is enough to play around!

## Brief description of this repository 
- `app` folder:
- `drop_here_and_see_your_pred` folder:
- `evaluation`:
    - kkkk
- `models`:
     - `mmm`:
     - `ooo`:
- `DIY_top_module_file.py`
- `Dockerfile`
- `final_tests.py`
- `get_your_toy_cgm_file.py`
- `LibreView_exp_launcher.py`
- `libreview_utils.py`
- `main_libreview.py`
- `README.md`
- `requirements.txt`
- `results_analysis.py`
- `sensor_params.py`
- `training_configs.py`
- `utils.py`
- `your_AI_DIY_parameters.py`

  ################### PENDING DESCRIPTIONNNNNN ########

### Try and run the experiments 

```
python exp_launcher.py loss_functions_comparison --model_hyperparameters KERNEL_SIZE STRIDE --training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE
```
For more information: 
```
python exp_launcher.py --help
```

### Time to play! Do you want to change the architectures, or include new models?

### Dockerization of the top module to make the "app" 

## Something missing? Need help? Any bugs? 
