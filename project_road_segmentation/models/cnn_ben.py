import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add

IMAGE_SIZE = 400

def cnn_ben(img_size = IMAGE_SIZE):
    kernel_size = (3,3)
    first_kernel_size = (5,5)
    dropout=0.2
    leakyrate=0.05
    model = tf.keras.models.Sequential([# Note the input shape is the desired size of the image 200x200 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, first_kernel_size, activation=LeakyReLU(alpha=leakyrate), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), padding='same'),
    tf.keras.layers.Conv2D(128, kernel_size, activation=LeakyReLU(alpha=leakyrate), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(128, kernel_size, activation=LeakyReLU(alpha=leakyrate), padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    Dropout(dropout),
    # The third convolutiont
    tf.keras.layers.Conv2D(256, kernel_size, activation=LeakyReLU(alpha=leakyrate), padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    Dropout(dropout),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, kernel_size, activation=LeakyReLU(alpha=leakyrate), padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    Dropout(dropout),
    tf.keras.layers.Conv2D(1, kernel_size, activation='sigmoid', padding='same'),])
    return model