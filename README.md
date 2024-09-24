# An AI-based "Do-it-Yourself" tool for personalized glucose prediction. :mechanical_arm::robot: 
Hi! :wave: :smile:  If you got here, it is very likely that you are interested in personalized glucose monitoring, AI, or both! If yours is the first case and you are willing to use this tool to predict your interstitial glucose level with your personalized AI model, [go directly to the *Use me!* section](#Use-me-!). If, on the contrary, you are more interested in the development of this tool :desktop_computer::keyboard:, the code, or you want to test your own AI models in this framework, you better go to [the developers and researchers section](#for-developers-and-researchers):man_technologist::woman_technologist: . 

### :thinking: Why _"Do-It-Yourself"_? 
The main characteristic of this AI-based module is that is _user-driven_. The user is the one to choose if she/he wants a personalized model or not, when to generate such model, and when to be prompted with a personalized prediction. Besides, it works with just the individual's data. Only __your data__ will be used to generate __your model__. Besides, __everything is LOCAL__. Your data, your model and your predictions will be accessed __only by you__ at your PC. No third-parties will participate on this process, only you!

This module has been developed with data from people with **type 1 diabetes**, but of course it is suitable for anyone who has a diabetes-related condition that implies having a glucose sensor attached to your body. :+1: 

If you are interested to read about the scientific basis of this work, please check our paper [*"An AI-Based “Do-It-Yourself” Module for Interstitial Glucose Forecasting for People with Type 1 Diabetes"*](https://ieeexplore.ieee.org/document/9851514) published on [npj Digital Medicine](https://www.nature.com/npjdigitalmed/) scientific journal. Besides, if this work somehow helped you with your research work or with a personal project, please, cite our paper. :page_with_curl:  

## Use me! 
### :handshake: First time? 
Welcome! First times are always special, and this is not an exception! :star_struck: The first time you use this module, you will get your personalized AI-glucose predictor. Once you have generated your model, you just have to follow the instructions on [Use me (again)! section](#Use-me-(again)!). Currently, this tool only supports the following sensors: 

#### Included sensors
| Data source  | Sensors | Input data | 
| ------------- | ------------- | ------------- |
| LibreView app | FreeStyle Libre | Glucose | 
| LibreView app | FreeStyle LibreLink | Glucose | 
| LibreView app | LibreLink | Glucose | 

:point_up_2: At this point, we assume that you have access to your real CGM data. If not, type in the terminal `python get_your_toy_cgm_file.py` to generate a LibreView-like CGM `.csv` file and play with this module! Same if your sensor is not included in the previous table (we're sorry, we are working on it! :hammer_and_wrench:) :bangbang:Of course, to run this script, you need to have Python installed on your PC:bangbang:

#### What do you need?
We have designed this module to be the least overwhelming possible to you. To install and use this module you just need a PC :computer:, an open terminal on it, an the [Docker Desktop installed in your PC](https://docs.docker.com/desktop/):cd:. Once you have downloaded it, you can proceed with the next steps. If everything goes as it should and as far as :warning:**you don't replace your CGM sensor**:warning:, you only will have to perform these steps **just ONCE**: 

**1)** Clone/Download this repository :octocat: (Obvious, but just in case!)

**2)** Open your Docker Desktop (if you are not in an admin account, right click and click on "Run as administrator").

**3)** Open a terminal. For example, typing "cmd" from the "Search" bar in Windows.

**4)** In the terminal, go to the directory where you want to install and save this module that will generate your personalized AI-model. Let's assume the directory `C:\Users\aralmeida`. Type or copy-paste in your terminal the following:

```
cd C:\Users\aralmeida
```
   
**5)** Now you are in your directory. Let's build the Docker image (basically, the "application", [here you have more information about it)](https://docs.docker.com/get-started/) to generate and execute your model later! Copy and paste the following line (the final dot is not a mistake!). After a few minutes, all you need to have your personalized AI model has been installed. 

```
docker build -t diy_cgm_image .
```

**6)** **IMPORTANT** :bangbang:: Create a folder :file_folder: named `/drop_your_data_here_and_see_your_pred`. Place it whenever you want in your PC. This folder is the one that will always be accesed by this app to execute the model and perform your prediction. :no_entry_sign:**DO NOT MOVE, REMOVE, OR CHANGE THE NAME OF THE FOLDER**:no_entry_sign: If you do so, you will have to recreate all this steps with the new directory :repeat_one:.

**7)**  Now, drop the file with your CGM data (usually ended with `.csv`) in the recently created `/drop_your_data_here_and_see_your_pred` :file_folder: folder.

**8)**  We are now ready to execute this app for the first time. Let's assume that the path where you placed your `/drop_your_data_here_and_see_your_pred` :file_folder:folder is `C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred`. As you can see, I created my folder just in the common `Downloads` folder in Windows. Now, you have to type (or copy/paste) this in your terminal **with your own directory, of course :bangbang:**:

```
docker run -ti --mount type=bind,source=/C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred,target=/CGM_forecasting/drop_your_data_here_and_see_your_pred diy_cgm_image
```

:warning:**WARNING**:warning:: if you, let's say, change the folder (for example) from `C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred` to `C/Users/aralmeida/Desktop/drop_your_data_here_and_see_your_pred`, you should replace `source=/C/Users/aralmeida/Downloads/drop_your_data_here_and_see_your_pred`by `source=/C/Users/aralmeida/Desktop/drop_your_data_here_and_see_your_pred`. Otherwise, you will recieve an error telling you that that directory does not exist!  

**9)** If everything went good, now you should see some interactive stuff on your terminal. You will be asked about the model of your CGM sensor, the glucose concentration units you are used to, or if you want to know a little bit more about the relation of your CGM data and your AI model. At this point, two different things could happen: 
   -  :white_check_mark:**THE IDEAL CASE**: the year that the app extracted from the file that you provided and will be used to generate and train your AI model do not contain too many interruptions an this app was able to generate a reliable AI-based predictor.
   -  :x::pensive:**BAD NEWS**: the year extracted from your uploaded CGM data contains too many interruptions to reliably generate an AI model. This is mainly due to sensor disconnections, misfunctions... We are sorry! The good news is that you can always go to the first step when you have more data available! We'll be waiting for you! :hugs:

**10)** Let's assume the best scenario. You were able to generate your AI model (after a few hours... :lotus_position_man:). If you go to your `/drop_your_data_here_and_see_your_pred` folder you will see a few things. You just have to focus on two.
   - `your_AI_based_CGM_predictor.h5`:robot: This is **your personalized AI model for your glucose level predictions!** Isn't it great?? __Do not delete this file__. If you do so, you will have to repeat all this process from the beginning.
   - `your_last_prediction.png`:chart_with_upwards_trend: This is your very first AI-based prediction! Of course, this prediction is not useful, since it took place after the model generation, that took a couple of hours (:lotus_position_man:). As you can see, the model takes the data from your last **24 hours**, and performs a prediction of your next **30 minutes**. The red shadow shows the error on each instant (15' and 30' ahead), that has been calculated during the model generation. From now on, everytime you execute this app, this picture will be overwritten with your most recent prediction, so this is what you will see! Unfortunately, if your last day contains CGM reading interruptions, the model will not be able to provide you with a prediction. This is (also) work in progress :hammer_and_wrench:.

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

##:man_technologist::woman_technologist:For developers and researchers

Hi again! If you are here, it is assumed that you have basic knowledge of Python programming and AI, so let's get into it!

The framework to design this **_"Do-It-Yourself" AI based module_** is based on Python 3.10 and TensorFlow 2.6.0. Typical libraries for Machine Learning development, such as `scikit-learn`, or `matplotlib` for visualization were also used. All the requirements are included in the `requirement.txt` file. Notice that if you don't use GPU, cuda libraries won't be used. 

If you want to explore the code and/or play with it, introduce your own models, change the data generation parameters, etc., your are in the right place! :basecampy:

**IMPORTANT :bangbang:** Go to the [TensorFlow website](https://www.tensorflow.org/install/source_windows?hl=es-419#gpu) to check the compatibilities between Python, CUDA, cuDNN and Tensorflow versions. 

After cloning this repository, go to the main folder and, first of all, create a conda environment with the Python 3.10 version.

```
conda create -n DIY_for_CGM_pred python=3.10
```
Activate your recently created environment

```
conda activate DIY_for_CGM_pred
````

Then, install all the required libraries. 
```
pip install --no-cache-dir -r requirements.txt 
```
Now, before running any experiments, you would need some CGM data. If you don't have any `.csv` file directly downloaded from the LibreView app, which could happen, execute the following script: 

```
python get_your_toy_cgm_file.py
```
In your current directory, you should now see a file with a name following the format that imitates LibreView-generated files: `IDXXX_SXXX_RXXX_glucose_DD-MM-YYY.csv`. Inside, there are fake CGM readings that **do not follow any CGM patterns**, just random numbers between the physiological glucose level limits. This data is in 'mg/dL'. Before running any script, drop your `.csv` files in a empty folder. 

Now, you are able to run experiments. It is desirable for you to get access to **real CGM data to get real conclusions** from your experiments. But, in the meantime, this is enough to play around! :juggling_person:

## Brief description of this repository 

__The most interesting files and folders to play, modify, and/or add stuff, such as DL models, are indicated with__ :rotating_light:. 

- :file_folder:`app` - Python functions encapsulated by the Dockerfile to exectute this module. Adapted from the rest of the files from this repository. Not useful at experimentation level.  
- :file_folder:`drop_here_and_see_your_pred` - Empty folder to map it to the Docker image to an actual folder in your host. Leave it as it is!
- :file_folder:`evaluation` - Multi- and single-step evalutation functions. **TO ADD MORE METRICS**, go to `model_evaluation()` function in the multi-step `evaluation.py` add them there. 
- :file_folder::rotating_light:`models` - TensorFloew descriptions of the single- and multi- step models are placed here. 'LSTM.py' is the one implemented in this app. In this work, multi-step has been implemented because it showed better performance. **TO ADD MORE MODELS**, create a new '.py' file inside this folder.
     - :page_facing_up::rotating_light:`training.py` - File containing key functions for this framework, like the description of the _ISO-adapted loss function_ and the _trimester-wise data partition_ (more details in [our paper]()).
- :page_facing_up:`DIY_top_module_file.py`- Application wrapper that is included in the Docker image. You can change the message prompting as you wish, but no much experimentation is expected here. 
- :page_facing_up:`Dockerfile`
- :page_facing_up:`final_tests.py` - Functions to perform a final test training the models with the whole year of data and testing it with additional collected data (check [our paper]() for more!).
- :page_facing_up:`get_your_toy_cgm_file.py` - Script to obtain a LibreView-alike '.csv' file containing CGM readings. 
- :page_facing_up:`LibreView_exp_launcher.py` - Script that allows the execution of this experiments from the terminal. 
- :page_facing_up:`libreview_utils.py` - Contains function for LibreView data curation and preparation. 
- :page_facing_up::rotating_light:`main_libreview.py` - The experiment itself. Check the following sections to know what to change. 
- :page_facing_up:`README.md` - You are now reading it! :hand_over_mouth:
- :page_facing_up:`requirements.txt` - All libraries required in your Python environment. Generated with `pip freeze > requirements.txt ` Some stuff is commented to avoid errors. 
- :page_facing_up:`results_analysis.py`- Functions to store results in different formats, and to generate graphs for data analysis and experiments comparison. 
- :page_facing_up::rotating_light:`sensor_params.py` - This file contains dictionaries containing the sensor information needed to generate the datasets, models and their evaluation. **TO ADD A SENSOR** create a dictionary here. The framework is parametrized, so no further changes are required. 
- :page_facing_up::rotating_light:`training_configs.py` - Training configurations to run the experiments. As with the sensors, create your own dictionary. Now, you have to add some lines in the `LibreView_exp_launcher.py`
- :page_facing_up:`utils.py` - Functions for data curation, dataset generation, or weights generation. 
- :page_facing_up::rotating_light:`your_AI_DIY_parameters.py` - Contains dictionaries to test your favourite configuration in the execution of the app. This is called by 'DIY_top_module.py'.

### Try and run the experiments 

For a first approach, it is good practice to just run the default experiments. See the command line to copy-paste to your terminal: 
   - `loss_function_comparison` : Dictionary in training_config.py. It contains models to test, input window length, normalization, data partition, etc.
   - `{1,2}` : Input features to use in the model. Currently 1 means only CGM, and 2 means CGM and its derivative. 
   - `{True, False}` : Weight the samples (hypo, hyper or normal range) in the training process.
   - `KERNEL_SIZE` : `10` (here you have to type a number. Otherwise, the default values in `LibreView_exp_launcher.py` will be taken. Same with the rest of the introduced parameters)
   - `STRIDE` : `1`
   - `EPOCHS`: `20`
   - `BATCH_SIZE` : `1`
   - `LEARNING_RATE`: `0.0001`

:rotating_light: :rotating_light:__CURRENT ISSUES TO BE SOLVED FOR CORRECT EXPERIMENT REPRODUCIBILITY (ADAPT YOUR CODE BEFORE RUNNING)__  :rotating_light: :rotating_light:
 - `set_of_libreview_keys` in `libreview_utils.py` is harcoded with the data we had to develop this tool. Now, replace this list by your keys. Having a file named `IDXXX_SYYY_RZZZ_glucose_DD-MM-YYY.csv`, the list entry would be `["XXX", "YYY", "ZZZZ", "DD-MM-YYYY"]`.
 - Directories are prepared for Windows, so check them if you are working on Linux. 

After this, you can run the big experiments with your models, custom configuration and sensors. 

```
python LibreView_exp_launcher.py LibreView-exp-1 2 True --model_hyperparameters 10 1 --training_hyperparameters 20 1 0.0001 --dataset_folder "C:\Users\aralmeida\Downloads\just_playing" # Replace the dataset folder
```
For more information about the arguments of this function: 
```
python exp_launcher.py --help
```

After the execution of the experiments, you will see many files and folders created, since this is thought to run a big number of experients with different variantes of the same parameter, as you can see in the code. For every validation fold, you have a dedicated `/evaluation` folder with some results graphs. For every included individual on the experiment there is a folder with the subject's `id`. For each `id`, there is a `results_dictionary.json`file with all the results. From this, results can be studied and compared within trainig strategies, subjects, DL models, etc. Now it's the moment to try your own experiments! (See further details in the functions and code documentation).  

### Time to play! Do you want to change the architectures, or include new models?
Let's begin with the interesting part! Let's assume that you have several `.csv` files containing __real__ CGM data. Otherwise, there is not much sense in running these experiments. Let's also assume that your files are also downloaded from the LibreView app. Otherwise, the functions such as `get_oldest_year_npys_from_LibreView_csv()`, or `prepare_LibreView_data()` __MUST BE REPLACED__. If this is your case, you should spend a while coding these functions to fit your files to the DL framework. Once this is done, you can proceed with the experimentation. We will skip things like normalization techniques, o cross-validation implementation. But those changes are straightworward by adding a different case in the `if else` sentences within the `main_libreview.py` file.  

#### 1. Your sensor
Declare a new dictionary inside `sensor_params.py` that contains the data from your sensor.  

   ```
   your_new_glucose_sensor = {
       "NAME" : "Your sensor",
       "CGM" : True,
       "INSULIN" : False, # This framework does not currently support insulin data 
       "SAMPLE_PERIOD" : 15, # Minutes between consecutive readings
       }
   ```

#### 2. Your new DL model.
   1) Go to `/models/multi_step` folder and create a `.py` file containing the TensorFlow description of your model. For example, `my_new_model.py`. Copy and paste the function structure from one of the other files contained in this folder.
   2) Import the model in `main_libreview.py` as it is done for the rest of the models.
   3) Include a new case in the `if else` sentence (around line 250 in `main_libreview.py`), with a recognizable string. For example `"my_new_model"`. Keep consistency with this identifier in the next steps. 

#### 3. Your training configuration.
Of course, there are model hyperparameters that are not adjustable for different DL models (LSTMs do not have kernels, and CNNs do not have a forget gate, for example), but some architectural designs can be fixed for all models for a more fair comparison.

1) Go to the `training_config.py` and add a new dictionary with your desired trainig configuration.

   ```
   your_new_training_config = {'sensor' : [your_new_glucose_sensor],
                   'N' : [96], # Input window length (number of CGM samples)
                   'step' : [1], # Step from one sample to the next one to generate the dataset to train the models
                   'PH' : [30, 60], # Prediction Horizon (how far ahead the model predicts) 
                   'single_multi_step' : ['multi'], # Single step or multi step prediction
                   'partition' : ['month-wise-4-folds'], # Data partition. See more details on our paper or in month_wise_LibreView_4fold_cv() in models/training.py
                   'normalization' : ['min-max'], # Data normalization
                   'under_over_sampling' : [None],  # Under- of Oversampling
                   'model' : ['my_new_model'], # Models to test. Can be more than one, as far as they are properly included within the code. 
                   'loss_function' : ['root_mean_squared_error'], # Currently MSE and our in-house designed ISO-adapted loss functions are supported
                   } 
   ```

2) Now that you have filled your dictionary, the experiment launcher should recognized the string associated with the dictionary of your new configuration, so you have to go to `LibreView_exp_launcher.py`and add the case in the `if else` sentence of `args.training_config`. Name it as in the dictionary `your_new_training_config`.  

#### 4. Run the script with your desired configuration
Don't worry if parameters like `KERNEL SIZE` do not apply to your model. They will be ignored. Let's launch a training with 2 inputs (some bugs pending with single input), and without weighting the samples. Kernel size and stride are 1 both. Finally, we will run 10 epochs, with a batch size of 2 and a learning rate of 0.0001. 

```
python LibreView_exp_launcher.py your_new_training_config 2 False --model_hyperparameters 1 1 --training_hyperparameters 10 2 0.0001 --dataset_folder "C:\Users\aralmeida\Downloads\just_playing"
```

Perfect! Now you have results from your experiments and you can compare within your models, or [against ours]()! Check the `results_analysis.py` functions and play a little bit to visualize your results from a Jupyter Notebook, for example! 

## Something missing? Need help? Any bugs? Suggestions?  

:email: antorguez95@hotmail.com
