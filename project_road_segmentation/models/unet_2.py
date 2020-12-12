import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add

IMAGE_SIZE = 400


def unet_2(n_filters=16, dropout=0.1, batchnorm=True, img_size=IMAGE_SIZE, kernel_size=3):
    input_img = Input((img_size, img_size, 3), name='img')
    # Contracting Path
    pows = [1, 2, 4, 8]
    last_pow = 16
    intermediaries = []
    p = input_img
    for pow in pows:
        c, p = down_block(p, n_filters * pow, dropout, batchnorm, kernel_size)
        intermediaries.append(c)

    # Expansive Path
    u = conv2d_block(p, n_filters * last_pow, kernel_size=kernel_size, batchnorm=batchnorm)
    for c, pow in zip(reversed(intermediaries), reversed(pows)):
        u = up_block(c, u, n_filters * pow, dropout, batchnorm, kernel_size)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def down_block(input, n_filters, dropout, batchnorm, kernel_size):
    c = conv2d_block(input, n_filters, kernel_size=kernel_size, batchnorm=batchnorm)
    p = MaxPooling2D((2, 2))(c)
    return c, Dropout(dropout)(p)


def up_block(concat, input, n_filters, dropout, batchnorm, kernel_size):
    u = Conv2DTranspose(n_filters, (kernel_size, kernel_size), strides=(2, 2), padding='same')(input)
    u = concatenate([u, concat])
    u = Dropout(dropout)(u)
    return conv2d_block(u, n_filters, kernel_size=kernel_size, batchnorm=batchnorm)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    x = half_conv2d_block(input_tensor, n_filters, kernel_size, batchnorm)
    return half_conv2d_block(x, n_filters, kernel_size, batchnorm)


def half_conv2d_block(input, n_filters, kernel_size, batchnorm):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    return Activation('relu')(x)
