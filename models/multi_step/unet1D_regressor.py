# Copyright (C) 2023 Antonio Rodriguez
# 
# This file is part of T1DM_WARIFA.
# 
# T1DM_WARIFA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# T1DM_WARIFA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with T1DM_WARIFA.  If not, see <http://www.gnu.org/licenses/>.

from tensorflow.keras import layers, Input, Model
import tensorflow as tf
from typing import Tuple, Dict

from arch_params import *

def encoding_block(x: tf.Tensor, filters: int, kernel_size: int, stride: int,
                   activation: str, padding: str, name_prefix: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encoding block of the network, based on the description of [1].

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
    
    # Max pooling to downsample in time
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same",
                            name=name_prefix+'_maxpool')(x) 

    return x, enc_residual

def decoding_block(x: tf.Tensor, residual: tf.Tensor, filters: int,
                   kernel_size: int, stride: int, activation: str, padding: str,
                   name_prefix: str) -> tf.Tensor:
    """Decoding block of the network, based on the description of [1].

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

    # Upsampling + 1 x ( Conv + ReLU )
    x = layers.UpSampling1D(size=2, name=name_prefix+'_upsample')(x)
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
def get_model(sensor : Dict, N: int = CGM_INPUT_POINTS, input_features: int = NUMBER_OF_INPUT_SIGNALS,
              tau : int = 1, kernel_size : int = 3, PH : int = 5) -> Model:
    
    """Returns a multistep regression model based on the 1D-UNET described in [1]. Some modifications 
    have been performed to adapt a segmentation model to a regresison model: activation functions,
    output dimension and the time distributed layers

    Args:
    -----
        N (int): Number of samples in the input tensor. Must be multiple of 2. Default: CGM_INPUT_POINTS.
        input_features (int): Number of features in the input tensor. Default: NUMBER_OF_INPUT_SIGNALS.
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
    x, res_1 = encoding_block(input, filters=input_features*2, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "enc_0")
    x, res_2 = encoding_block(x, filters=input_features*4, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "enc_1")
    x, res_3 = encoding_block(x, filters=input_features*8, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "enc_2")
    x, res_4 = encoding_block(x, filters=input_features*16, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "enc_3")

    # Intermediate layer 
    # 2 x ( Conv + ReLU )
    x = layers.Conv1D(filters=input_features*32, kernel_size=kernel_size, strides=1, activation="linear", padding="same", name='central_conv_relu_0')(x)
    x = layers.Conv1D(filters=input_features*32, kernel_size=kernel_size, strides=1, activation="linear", padding="same", name='central_conv_relu_1')(x)
       
    # Decoding phase of the network: upsampling inputs
    x = decoding_block(x, res_4, filters=input_features*16, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "dec_0")
    x = decoding_block(x, res_3, filters=input_features*8, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "dec_1")
    x = decoding_block(x, res_2, filters=input_features*4, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "dec_2")
    x = decoding_block(x, res_1, filters=input_features*2, kernel_size=kernel_size, stride = tau, activation = "linear", padding = "same", name_prefix = "dec_3")

    # Output of the model (modified from [1] to switch from classification to regression) 
    x = layers.Conv1D(filters=input_features, kernel_size=kernel_size, strides=tau, activation="sigmoid", padding="same", name='final_conv')(x)

    # Reshape x to be a 3D tensor
    x = layers.Reshape((input_features, N), input_shape=(N, input_features))(x)

    # Add timeDistributed dense layers
    x = layers.TimeDistributed(layers.Dense(32))(x)

    # Once flattened, add a dense layer to predict the output
    output = layers.Dense(PH/sensor["SAMPLE_PERIOD"])(x) # PH/SENSOR_SAMPLING_FREQUENCY

    # Define the model
    model = Model(input, output)


    return model