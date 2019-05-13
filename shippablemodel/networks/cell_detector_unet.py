from typing import Tuple

from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model


def conv_dropout_conv(previous_layer, kernel_size, stride, dropout_rate):
    layer = Conv2D(kernel_size, stride, activation='elu', kernel_initializer='he_normal', padding='same')(previous_layer)
    layer = Dropout(dropout_rate)(layer)
    return Conv2D(kernel_size, stride, activation='elu', kernel_initializer='he_normal', padding='same')(layer)


def unet(_input_shape: Tuple[int, ...]) -> Model:
    inputs = Input(_input_shape)
    normalized_inputs = Lambda(lambda x: x / 255)(inputs)
    cdc1 = conv_dropout_conv(normalized_inputs, 16, (5, 5), 0.1)
    pool1 = MaxPooling2D((2, 2))(cdc1)
    cdc2 = conv_dropout_conv(pool1, 32, (5, 5), 0.1)
    pool2 = MaxPooling2D((2, 2))(cdc2)
    cdc3 = conv_dropout_conv(pool2, 64, (3, 3), 0.2)
    pool3 = MaxPooling2D((2, 2))(cdc3)
    cdc4 = conv_dropout_conv(pool3, 128, (3, 3), 0.2)
    pool4 = MaxPooling2D((2, 2))(cdc4)
    cdc5 = conv_dropout_conv(pool4, 256, (3, 3), 0.3)
    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(cdc5), cdc4])
    cdc6 = conv_dropout_conv(up6, 256, (3, 3), 0.3)
    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(cdc6), cdc3])
    cdc7 = conv_dropout_conv(up7, 64, (3, 3), 0.2)
    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(cdc7), cdc2])
    cdc8 = conv_dropout_conv(up8, 32, (3, 3), 0.1)
    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(cdc8), cdc1], axis=3)
    cdc9 = conv_dropout_conv(up9, 16, (3, 3), 0.1)
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(cdc9)

    model = Model(inputs=[inputs], outputs=[output_layer])
    return model

