# An AI-based "Do-it-Yourself" tool for personalized glucose prediction. :mechanical_arm::robot: 
Hi! :wave: :smile:  If you got here, it is very likely that you are interested in personalized glucose monitoring, AI, or both! If it is the first case and you are willing to use this tool to predict your interstitial glucose level with your personalized AI model, [go directly to the *Use me!* section](#Use-me-!). If, on the contrary, you are more interested in the development of this tool :desktop_computer::keyboard:, the code, or you want to test your own AI models in this framework, you better go to [the developers and researchers section](#For-developers-and-researchers) :man_technologist::woman_technologist:. 

### Why _"Do-It-Yourself"_? :thinking:
The main characteristic of this AI-based module is that is _user-driven_. She/He is the one to choose if they want a personalized model or not, when to generate such model, and when to be prompted with a personalized prediction. Besides, it works with just the individual's data. Only __your data__ will be used to generate __your model__. Besides, __everything is LOCAL__. Your data, your model and your predictions will be accessed only by you at your PC. No third-parties will take part on this process, only you!

This module has been developed with data from people with **type 1 diabetes**, but of course it is suitable for anyone who has a diabetes-related condition that implies having a glucose sensor attached to your body. :+1: 

If you are interested to read about the scientific basis of this work, please check our paper [*"An AI-Based “Do-It-Yourself” Module for Interstitial Glucose Forecasting for People with Type 1 Diabetes"*](https://ieeexplore.ieee.org/document/9851514) published on [npj Digital Medicine](https://www.nature.com/npjdigitalmed/) scientific journal. Besides, if this work somehow helped you with your research work or with a personal project, please, cite our paper. :page_with_curl:  

## Use me!  
### First time? :handshake:
Welcome! First time is always special, and this is not an exception. The first time you use this module, you will get your personalized AI-glucose predictor. Once you have generated your model, you just have to follow the instructions on [Use me (again)! section](#Use-me-(again)!). Currently, this tool only supports the following sensors: 

#### Included sensors
| Data source  | Sensors | Input data | 
| ------------- | ------------- | ------------- |
| LibreView app | FreeStyle Libre | Glucose | 
| LibreView app | FreeStyle LibreLink | Glucose | 
| LibreView app | LibreLink | Glucose | 

:point_up_2: At this point, we assume that you have access to your real CGM data. If not, type in the terminal `python get_your_toy_cgm_file.py` to generate a LibreView-like CGM `.csv` file and play with this module! Same if your sensor is not included in the previous table (we're sorry, we are working on it! :hammer_and_wrench:) :bangbang:Of course, to run this script, you need to have Python installed on your PC:bangbang:

#### What do you need?
We have designed this module to be the least overwhelming possible to you. So, to install and execute this module you just need a PC :computer:, an open terminal, an the [Docker Desktop installed in your PC](https://docs.docker.com/desktop/):cd:. Once you have downloaded it, you can proceed with the next steps. If everything goes as it should and as far as :warning:**you don't replace your CGM sensor**:warning:, you only will have to perform these steps **just ONCE**: 

**1)** Open your Docker Desktop (if you are not in an admin account, right click and click on "Run as administrator").

**2)** Open a terminal. For example, typing "cmd" from the "Search" bar in Windows.

**3)** In the terminal, go to the directory where you want to install and save this module that will generate your personalized AI-model. Let's assume the directory `C:\Users\aralmeida`. Type or copy-paste in your terminal the following:

```
cd C:\Users\aralmeida
```
   
**4)** Now you are in your directory. Let's build the Docker image [(go here if you are interested on this)](https://docs.docker.com/get-started/) to generate and execute your model later! Copy and paste the following line (the final dot is not a mistake!). After a few minutes, all you need to have your personalized AI model has been installed. 

```
docker build -t diy_cgm_image .
```

**5)** **IMPORTANT** :bangbang:: Create a folder :file_folder: named `/drop_your_data_here_and_see_your_pred`. Place it whenever you want in your PC. This folder is the one that will always be accesed by this app to execute the model and perform your prediction. :no_entry_sign:**DO NOT MOVE, REMOVE, OR CHANGE THE NAME OF THE FOLDER**:no_entry_sign: If you do so, you will have to recreate all this steps with the new directory :repeat_one:.

**6)**  Now, drop the file with your CGM data (usually ended with `.csv`) in the recently created `/drop_your_data_here_and_see_your_pred` :file_folder: folder.

**7)**  We are now ready to execute this app for the first time. Let's assume that the path where you placed your `/drop_your_data_here_and_see_your_pred` :file_folder:folder is `C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred`. As you can see, I created my folder just in the common `Downlands` folder in Windows. Now, you have to type (or copy/paste) this in your terminal **with your own directory, of course :bangbang:**:

```
docker run -ti --mount type=bind,source=/C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred,target=/CGM_forecasting/drop_your_data_here_and_see_your_pred diy_cgm_image
```

:warning:**WARNING**:warning:: if you, let's say, change the folder (for example) from `C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred` to `C/Users/aralmeida/Desktop/drop_your_data_here_and_see_your_pred`, you should replace `source=/C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred`by `source=/C/Users/aralmeida/Desktop/drop_your_data_here_and_see_your_pred`. Otherwise, you will recieve an error telling you that that directory does not exist!  

**8)** If everything went good, now you should see some interactive stuff on your terminal. You will be asked about the model of your CGM sensor, the glucose concentration units you are used to, or if you want to know a little bit more about the relation of your CGM data and your AI model. At this point, two different things could happen: 
   -  :white_check_mark:**THE IDEAL CASE**: the year that the app extracted from the file that you provided and will be used to generate and train your AI model do not contain too many interruptions an this app was able to generate a reliable AI-based predictor.
   -  :x::pensive:**BAD NEWS**: the year extracted from your uploaded CGM data contains too many interruptions to reliably generate an AI model. This is mainly due to sensor disconnections, misfunctions... We are sorry! The good news is that you can always go to the first step when you have more data available! We'll be waiting for you! :hugs:

**9)** Let's assume the best scenario. You were able to generate your AI model (after a few hours... :lotus_position_man:). If you go to your `/drop_your_data_here_and_see_your_pred` folder you will see a few things. You just have to focus on two.
   - `your_AI_based_CGM_predictor.h5`:robot: This is **your personalized AI model for your glucose level predictions!** Isn't it great?? Please, do not delete this file. If you do so, you will have to repeat all this process from the beginning.
   - `your_last_prediction.png`:chart_with_upwards_trend: This is your very first AI-based prediction! Of course, this prediction is not useful, since it took place after the model generation, that took a couple of hours (:lotus_position_man:). As you can see, the model takes the data from your last **24 hours**, and performs a prediction of your next 30'. The red shadow shows the error on each instant (15' and 30' ahead), that has been calculated during the model generation. From now on, everytime you execute this app, this picture will be overwritten with your most recent prediction, so this is what you will see! Unfortunately, if your last day contains CGM reading interruptions, the model will not be able to provide you with a prediction. This is (also) work in progress :hammer_and_wrench:.

From now, everytime you want to use this app, you have to follow the simple steps described in [Use me (again)! section](#Use-me-(again)!). 

### Use me (again)! :repeat:

Hi again! If you are here, it means you have `your_AI_based_CGM_predictor.h5` file in your `/drop_your_data_here_and_see_your_pred` folder, and you want to get now a personalized prediction of your next 30 minutes. 
   1) :arrow_down: Download your most recent data in a `.csv` file, containing at least your last 24 hours of CGM readings. Don't worry if your downloaded file contains more than 24 hours, this app will take care of this!
   2) :file_folder: Drop it in the `/drop_your_data_here_and_see_your_pred` folder.
   3) :desktop_computer: Open a terminal in your PC and execute the following lines (remember to replace `APP_DIRECTORY` by the same as you used in step 3 of [Use me! section](#Use-me!) and `YOUR_DATA_DIRECTORY/drop_your_data_here_and_see_your_pred` by the directory where you save your CGM file):
   ```
   cd APP_DIRECTORY
   ```
   
   ```
   docker run -ti --mount type=bind,source=/YOUR_DATA_DIRECTORY/drop_your_data_here_and_see_your_pred,target=/CGM_forecasting/drop_your_data_here_and_see_your_pred diy_cgm_image
   ```
   4) :chart_with_upwards_trend: Now, the terminal will promtp you with some messages. If you had CGM readings interruptions in your last 24 hours, we cannot provide a reliable prediction. We are sorry! If your data was OK, please check the `your_last_prediction.png` picture with your updated prediction. And, please, never take this as an absolute truth! This is just an AI-based prediction!

From now on, these are the steps you have to follow any time you want to have a personalized glucose prediction based on your own AI model! Hope this helps in your day-to-day! 

## For developers and researchers :man_technologist::woman_technologist:
Hi again! If you are here, it is assumed that you have basic knowledge of Python programming and AI, so let's get into it!

The framework to design this _"Do-It-Yourself"_ AI based module is based on Python 3.10 and TensorFlow 2.6.0. Typical libraries for Machine Learning development, such as `scikit-learn`, or `matplotlib` for visualization. All the requirements are included in the `requirement.txt` file. Notice that if you don't use GPU, cuda libraries won't be used. So, if you want to explore the code and/or play with it, introduce your own models, change the data generation parameters, etc., your are in the right place! First of all, create a conda environment. 

**IMPORTANT :bangbang:** Go to the [TensorFlow website](https://www.tensorflow.org/install/source_windows?hl=es-419#gpu) to check the compatibilities between Python, CUDA, cuDNN and Tensorflow versions. 

After cloning this repository, go to the main folder and copy and paste the following:

```
conda create -n DIY_for_CGM_pred python=3.10
```
Then, install all the required libraries. 
```
pip install --no-cache-dir -r requirements.txt 
```
Now, you before running any experiments, you would need some CGM data. If you don't have any `.csv` file directly downloaded from the LibreView app, which could happen, execute the following script: 

```
python get_your_toy_cgm_file.py
```
In your current directory, you should now see a file with a name following the format that imitates LibreView-generated files: `IDXXX_SXXX_RXXX_glucose_DD-MM-YYY.csv`. Inside, there are fake CGM readings that do not follow any CGM patterns, just random numbers between the physiological glucose level limits. This data is in 'mg/dL'. 

Now, you are able to run experiments. It is desirable for you to get access to real CGM data to get real conclusion from this. But, in the meantime, this is enough to play around!

## Brief description of this repository 
- :file_folder:`app` :
- :file_folder:`drop_here_and_see_your_pred`:
- :file_folder:`evaluation`:
    - kkkk
- :file_folder:`models`:
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

:email:
