# Copyright (C) 2024 Antonio Rodriguez
# 
# This file is part of Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool.
# 
# Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Personalized-AI-Based-Do-It-Yourself-Glucose-Prediction-tool.  If not, see <http://www.gnu.org/licenses/>.

from tensorflow.keras import layers, Input, Model
import tensorflow as tf
from typing import Tuple, Dict

def encoding_block(x: tf.Tensor, filters: int, kernel_size: int, stride: int,
                   activation: str, padding: str, name_prefix: str) -> Tuple[tf.Tensor, tf.Tensor]:
    
    """Encoding block of the network, modified from [1]. Maxpooling layers has been 
    removed to remove the time compression from the network. 

    Args:
    -----
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters in the convolutional layers.
        kernel_size (int): Kernel size of the convolutional layers.
        stride (int): Stride of the convolutional layers.
        activation (str): Activation function of the convolutional layers.
        padding (str): Padding of the convolutional layers.
        name_prefix (str): Name prefix of the layers.
    
    Returns:
    --------
        x (tf.Tensor): Output tensor.
        enc_residual (tf.Tensor): Residual tensor to use in the decoder phase.
    
    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """
    # 2 x ( Conv + ReLU )
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_0')(x)

    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_1')(x)
    
    # Store tensor to be the input od the decoding phase
    enc_residual = x 
    
    return x, enc_residual

def decoding_block(x: tf.Tensor, residual: tf.Tensor, filters: int,
                   kernel_size: int, stride: int, activation: str, padding: str,
                   name_prefix: str) -> tf.Tensor:
    """Decoding block of the network, modified from [1]. Upsampling layer has been 
    removed, since compression from the encoding block was removed.

    Args:
    -----
        x (tf.Tensor): Input tensor.
        residual (tf.Tensor): Residual tensor from the encoder phase.
        filters (int): Number of filters in the convolutional layers.
        kernel_size (int): Kernel size of the convolutional layers.
        stride (int): Stride of the convolutional layers.
        activation (str): Activation function of the convolutional layers.
        padding (str): Padding of the convolutional layers.
        name_prefix (str): Name prefix of the layers.
    
    Returns:
    --------
        x (tf.Tensor): Output tensor.

    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """

    # 1 x ( Conv + ReLU )
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_up_conv_relu')(x)


    # Add residual + 2 x ( Conv + ReLU )
    x = layers.Concatenate(name=name_prefix+'_concatenate')([residual, x]) 
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_0')(x)

    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride,
                      activation=activation, padding=padding, use_bias=False,
                      name=name_prefix+'_conv_relu_1')(x)
      
    return x

# Returns a CNN-model instance 
def get_model(sensor : Dict, N: int, input_features: int = 1,
              tau : int = 1, kernel_size : int = 3, PH : int = 5) -> Model:
    """Returns a multi step regression model based on the 1D-UNET described in [1]. Some modifications 
    have been performed to adapt a segmentation model to a regression model: activation functions,
    output dimension and the time distributed layers. Furthermore, time compression and, consequently,
    decompression has been removed . The Prediction Horizon of the model is defined by the previously
    generated training dataset since it does not influence the model's architecture.

    Args:
    -----
        sensor (Dict) : Dictionary with the sensor's information, such as the sampling frequency.
        N (int): Number of samples in the input tensor. Must be multiple of 2.
        input_features (int): Number of features in the input tensor. Default: 1.
        tau (int): Stride of the convolutional layers. Default: 1, as [1]
        kernel_size (int): Kernel size of the convolutional layers. Default: 3, as [1]
        PH (int): Prediction Horizon to predict. Length of the predicted sequence lenght = PH/sampling frequency of
        the sensor. Default: 5.
    
    Returns:
    --------
        model (Model): CNN-model instance.
    
    References:
    -----------
        [1] F. Renna, J. Oliveira and M. T. Coimbra, "Deep Convolutional Neural
        Networks for Heart Sound Segmentation," in IEEE Journal of Biomedical
        and Health Informatics, vol. 23, no. 6, pp. 2435-2445, Nov. 2019, doi:
        10.1109/JBHI.2019.2894222.
    """
    # Input tensor
    input = Input(shape=(N, input_features)) 

    # Encoding phase of the network: downsampling inputs 
    x, res_1 = encoding_block(input, filters=input_features*2, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "enc_0")
    x, res_2 = encoding_block(x, filters=input_features*4, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "enc_1")
    x, res_3 = encoding_block(x, filters=input_features*8, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "enc_2")
    x, res_4 = encoding_block(x, filters=input_features*16, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "enc_3")

    # Intermediate layer 
    # 2 x ( Conv + ReLU )
    x = layers.Conv1D(filters=input_features*32, kernel_size=kernel_size, strides=1, activation="relu", padding="same", name='central_conv_relu_0')(x)
    x = layers.Conv1D(filters=input_features*32, kernel_size=kernel_size, strides=1, activation="relu", padding="same", name='central_conv_relu_1')(x)
       
    # Decoding phase of the network: upsampling inputs
    x = decoding_block(x, res_4, filters=input_features*16, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "dec_0")
    x = decoding_block(x, res_3, filters=input_features*8, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "dec_1")
    x = decoding_block(x, res_2, filters=input_features*4, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "dec_2")
    x = decoding_block(x, res_1, filters=input_features*2, kernel_size=kernel_size, stride = tau, activation = "relu", padding = "same", name_prefix = "dec_3")

    # Output of the model (modified from [1] to switch from classification to regression) 
    x = layers.Conv1D(filters=input_features, kernel_size=kernel_size, strides=tau, padding="same", name='final_conv')(x)

    # Reshape x to be a 3D tensor
    x = layers.Reshape((input_features, N), input_shape=(N, input_features))(x)

    # Add timeDistributed dense layers
    # x = layers.TimeDistributed(layers.Dense(N/2))(x)
    # x = layers.TimeDistributed(layers.Dense(N/4))(x)
    # x = layers.TimeDistributed(layers.Dense(8))(x)
    # x = layers.TimeDistributed(layers.Dense(2))(x)
    # x = layers.Dense(N/2)(x)
    # x = layers.Dense(N/4)(x)

    # Once flattened, add a dense layer to predict the output
    # output = layers.TimeDistributed(layers.Dense(PH/sensor["SAMPLE_PERIOD"]))(x)
    output = layers.Dense(PH/sensor["SAMPLE_PERIOD"])(x)

    # Define the model
    model = Model(input, output)


    return model