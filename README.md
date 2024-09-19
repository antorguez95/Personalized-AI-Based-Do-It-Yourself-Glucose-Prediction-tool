# An AI-based "Do-it-Yourself" tool for personalized glucose prediction.
Hi! If you got here, it is very likely that you are interested in blood glucose monitoring, AI, or both! If it is the first case and you are willing to use this tool to predict your interstitial glucose level with your personalized AI model, [go directly to the *Use me!* section](#Use-me-!). If, on the contrary, you are more interested in the development of this tool, the code, or you want to test your own models in this framework, for example, you better [go directly to the developers and researchers section](#For-developers-and-researchers). 


If you are interested to read about the scientific basis of this work, please check our paper [*"An AI-Based “Do-It-Yourself” Module for Interstitial Glucose Forecasting for People with Type 1 Diabetes"*](https://ieeexplore.ieee.org/document/9851514) published on [npj Digital Medicine]([https://www.nature.com/npjdigitalmed/]) scientific journal. Besides, if this work somehow helped you with your research work or with a personal project, please, cite out paper.  

## Use me!  
### First time?
First time is always special, and this is not an exception. The first time you use this module, you will get your personalized AI-glucose predictor. Once you have generated your model, you just have to follow the instructions on [Use me (again)! section](#Use-me-(again)!). Currently, this tool only supports the following sensors: 

#### Included sensors
| Data source  | Sensors | Input data | 
| ------------- | ------------- | ------------- |
| LibreView app | FreeStyle Libre | Glucose | 
| LibreView app | FreeStyle LibreLink | Glucose | 
| LibreView app | LibreLink | Glucose | 

At this point, we assume that you have access to your real CGM data. If not, type in the terminal `python get_your_toy_cgm_file.py` to generate a LibreView-like CGM `.csv` file and play with this module! Same if your sensor is not included in the previous table (we're sorry, we are working on it!). 

#### What do you need?
We have designed this module to be the least overwhelming possible to you. So, to install and execute this module you just need a PC, an open terminal, an the [Docker Desktop installed in your PC](https://docs.docker.com/desktop/). Once you have downloaded it, you can proceed with the next steps: 

- 1) Open your Docker Desktop (if you are not in an admin account, right click and clik on "Run as administrator").
- 2) Open a terminal. For example, typing "cmd" from the "Search" bar in Windows.
- 3) In the terminal, go to the directory where you want to install and save this module together with your personalized AI-model. Let's asume the directory `C:\Users\aralmeida`:
     `cd C:\Users\aralmeida`
- 4) Now you are in your directory. Let's build the Docker image [(go here if you are interested on this)](https://docs.docker.com/get-started/) to run your model later! Copy and paste the following (the final dot is not a mistake!). 
     `docker build -t diy_cgm_image . `
- 5) After a few minutes, all you need to have your personalized AI model has been installed. 
  


P.S.: even it seems pretty obvious, this tool is useful for people that has a glucose sensor to monitor its glucose levels! Even if you have T1DM, T2DM, gestational diabetes... Doesn't matter, this tool will suit you as far as you have a glucose sensor attach to your body! However, if you don't have any sensors... we cannot help you this time! Nonetheless, if you are curious, you can always check the tool if you this example data(____ link a datos que podamos usar publicamente). Maybe you can help us to improve this framework!

### Use me (again)!

#### Requirements 
__ de momento ni puta idea
#### Instalation process
__ de momento ni puta idea
#### First use
Now that you install all the necessary stuff, you are ready to deploy your own AI-based glucose level predictor! 
    
        1) Go to ___
        2) Upload your ´.csv´ file downladed from your 'LibreView' or 'Glooko' personal account. An analysis of the uploaded file will be automatically done. There are two options:
            a) The amount of data recorded by your sensor is enough to develop a fully-personalized model. (____ AÑADIR MINIMO DE DATOS, RECOMENDACION DE QUE ENVIEN + DEL MINIMO PORSIACA)
            b) The amount of data recorded by your sensor is too little to develop a enough reliable AI model. A global AI model is used, and you data is used to partially-personalized the model.
        3) After waiting some time (around 1 hour), your model will be ready to recieve data from your sensor and provide real-time 1-hour predictions, being updated with every reading of the sensor. 

        ____ IMAGES
        
### Do you want to update your model with more data?
If this is your first time using this tool, yo should go first to (____ENLACE SECCION). If you want to update your model with more data, welcome again! The more data you provide to your AI-model, the better will it be performing! So, if you want to update your AI-based glucose level predictor, please check out the following steps: 
        
        1) Go to ____
        2) Upload your '.csv' file downladed from your 'LibreView' or 'Glooko' personal account. An analysis of the uploaded file will be automatically done. There are two options, but these are different 
            a) 
            b) 
        3) After waiting for some time (depending of the amount of data uploaded to update the model, but it shouldn't take more than 15 minutes), your re-trained AI-predictor is ready to work again!

    ____ IMAGES


## For developers and researchers
Hi again! If you are here, it is assumed that you have basic knowledge of Python programming and AI, so let's get into it!

### Requirements 

`conda create -n T1DM_WARIFA python=3.10`

`conda install -c anaconda pandas`
`conda install -c conda-forge matplotlib`
`conda install -c conda-forge tensorflow` 
`pip install pydot`
`conda install -c anaconda graphviz`
`conda install -c conda-forge shapely`
`conda install -c anaconda scikit-learn`
`conda install -c anaconda openpyxl`
`conda install -c anaconda statsmodels`


### How to run the experiments 

`python exp_launcher.py loss_functions_comparison --model_hyperparameters KERNEL_SIZE STRIDE --training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE`

For more information, type `python exp_launcher.py --help`

## compatibilidades con entrenamiento tensorflow y gpu

Go to the [TensorFlow website](https://www.tensorflow.org/install/source_windows?hl=es-419#gpu) to check the compatibilities between Python, CUDA, cuDNN and Tensorflow versions for Windows.

`conda create -n T1DM_WARIFA_GPU python=3.10`
`conda activate T1DM_WARIFA_GPU`
`pip install tensorflow-gpu==2.6.0`

`conda install -c conda-forge xlsxwriter`

`conda install seaborn -c conda-forge`

## How to include your model in this framework: 
