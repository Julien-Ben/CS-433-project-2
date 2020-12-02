from tensorflow.keras import layers, models
from .constants import *


def unet_1():
    f = [16, 32, 64, 128, 256]
    inputs = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = models.Model(inputs, outputs)
    return model


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = layers.UpSampling2D((2, 2))(x)
    concat = layers.Concatenate()([us, skip])
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
