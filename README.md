# A Personalized AI-based "Do it Yourself!" tool for glucose prediction.
Hi! If you got here, it is very likely that you are interested in blood glucose monitoring, AI, or both! If it is the first case and you are willing to use our tool to track your glucose level in real time (meaning every 5-15 minutes), you can go directly to (___ENLACE A SECCION FOR USERS). If, on the contrary, you are more interested in the development of this tool, this code, or you want to test your own models in this framework, for example. you better go to (____ENLACE A FOR DEVELOPERS). 

## What's in this repository?

Our work enclosed in the WARIFA project :O
:) jaja

## Included sensors


| Data source  | Sensors |
| ------------- | ------------- |
| LibreView app  | FreeStyle Libre, LibreLink  |
| Glooko app  | Content Cell  |

Freestyle Libre
Glooko

## For users (Diabetes Mellitus patient or not) 
### First time? 
If it's your first time using this tool, here are the detailed steps you should follow: 
    
        1) Go to ___
        2) Upload your '.csv' file downladed from your 'LibreView' or 'Glooko' personal account. An analysis of the uploaded file will be automatically done. There are two options:
            a) The amount of data recorded by your sensor is enough to develop a fully-personalized model. (____ AÑADIR MINIMO DE DATOS, RECOMENDACION DE QUE ENVIEN + DEL MINIMO PORSIACA)
            b) The amount of data recorded by your sensor is too little to develop a enough reliable AI model. A global AI model is used, and you data is used to partially-personalized the model.
        3) After waiting some time (around 1 hour), your model will be ready to recieve data from your sensor and provide real-time 1-hour predictions, being updated with every reading of the sensor. 

        ____ IMAGES
        
### Do you want to upload your model with more data?
If this is your first time using this tool, yo should go first to (____ENLACE SECCION). If you want to update your model with more data, welcome again! The more data you provide to your AI-model, the better will it be performing! So, if you want to update your AI-based predictor, please follow the following steps: 
        1) Go to ____
        2) Upload your '.csv' file downladed from your 'LibreView' or 'Glooko' personal account. An analysis of the uploaded file will be automatically done. There are two options, but these are different 

### How can I use this tool?


## For developers
## Requirements 

`conda create -n T1DM_WARIFA python=3.10`

`conda install -c anaconda pandas`
`conda install -c conda-forge matplotlib`
`conda install -c conda-forge tensorflow` 
`pip install pydot`
`conda install -c anaconda graphviz`
`conda install -c conda-forge shapely`
`conda install -c anaconda scikit-learn`
`conda install -c anaconda openpyxl`

## How to run the experiments 

`python exp_launcher.py loss_functions_comparison --model_hyperparameters KERNEL_SIZE STRIDE --training_hyperparameters EPOCHS BATCH_SIZE LEARNING_RATE`

For more information, type `python exp_launcher.py --help`

## compatibilidades con entrenamiento tensorflow y gpu

Check https://www.tensorflow.org/install/source_windows?hl=es-419#gpu wor compatibilities between Python, CUDA, cuDNN and Tensorflow for Windows.

`conda create -n T1DM_WARIFA_GPU python=3.10`
`conda activate T1DM_WARIFA_GPU`
`pip install tensorflow-gpu==2.6.0`

conda install -c conda-forge xlsxwriter



## Input data 

Glucose
Insulin

## How to include your model in this framework: 
