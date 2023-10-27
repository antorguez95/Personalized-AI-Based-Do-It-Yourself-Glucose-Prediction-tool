# A Personalized AI-based "Do it Yourself!" for glucose prediction.
Hi! If you got here, it is very likely that you are interested in blood glucose monitoring, AI, or both! If it is the first case and you are willing to use our tool to track your glucose level in real time (meaning every 5-15 minutes), you can [go directly to](##For-users). If, on the contrary, you are more interested in the development of this tool, this code, or you want to test your own models in this framework, for example. you better [go to the developers section](##For-developers). 

If you are interested to go deep into this work, please check our paper(...) published in journal(...). Besides, if this work somehow helped you with your research work or with a personal project, please, cite out paper.  

## What's in this repository?

Our work enclosed in the [WARIFA project](https://www.warifa.eu/) 

For more technical details regarding the content of the repository folders, files, etc., please [go to the developers section](##For-developers). 

## For users (Diabetes Mellitus patient or not) 
### First time? 
If it's your first time using this tool, here are the detailed steps you should follow, beginning with the terminal requirements to install and run this tool, and with a brief guide to begin to train and run your own personalized AI-based glucose level predictor!  

P.S.: even it seems pretty obvious, this tool is useful for people that has a glucose sensor to monitor its glucose levels! Even if you have T1DM, T2DM, gestational diabetes... Doesn't matter, this tool will suit you as far as you have a glucose sensor attach to your body! However, if you don't have any sensors... we cannot help you this time! Nonetheless, if you are curious, you can always check the tool if you this example data(____ link a datos que podamos usar publicamente). Maybe you can help us to improve this framework!

So far, these are the platforms and the sensors associated to them that are supported by this tool

#### Included sensors
| Data source  | Sensors | Input data | 
| ------------- | ------------- | ------------- |
| LibreView app | FreeStyle Libre | Glucose | 
| Glooko app  | -  | Glucose, Insulin | 

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


## For developers
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

### How to run the experiments 

`python exp_launcher.py loss_functions_comparison --model_hyperparameters KERNEL_SIZE STRIDE --training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE`

For more information, type `python exp_launcher.py --help`

## compatibilidades con entrenamiento tensorflow y gpu

Go to the [TensorFlow website](https://www.tensorflow.org/install/source_windows?hl=es-419#gpu) to check the compatibilities between Python, CUDA, cuDNN and Tensorflow versions for Windows.

`conda create -n T1DM_WARIFA_GPU python=3.10`
`conda activate T1DM_WARIFA_GPU`
`pip install tensorflow-gpu==2.6.0`

`conda install -c conda-forge xlsxwriter`



## Input data 

Glucose
Insulin

## How to include your model in this framework: 
