# A Personalized AI-based "Do it Yourself!" tool for glucose prediction.
Hi! If you got here, it is very likely that you are interested in blood glucose monitoring, AI, or both! If it is the first case and you are willing to use a tool to track your glucose level in real time (meaning every 5-15 minutes), you can go directly to (___ENLACE A SECCION FOR USERS). If, on the contrary, you are more interested in the development of this tool, this code, or you want to test your own models in this framework, for example. you better go to (____ENLACE A FOR DEVELOPERS). 

## What's in this repository?

Our work enclosed in the WARIFA project :O
:) jaja

## For users (Diabetes Mellitus patient or not) 
In this section, you are assumed to be a user of this tool for the very first time. 

1) First: upload your .csv file
2) An analysis of the uploaded file is done. There are tu options:
    a) The amount of data you uploaded is enough

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

## Included sensors

Freestyle Libre

## Input data 

Glucose
Insulin

## How to include your model in this framework: 
