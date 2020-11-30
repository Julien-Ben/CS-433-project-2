import os
import sys
import gzip
import code
import urllib
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

sys.path.append('./')
from helpers.helpers import *
from helpers.mask_to_submission import *

COLAB = False
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
GENERATE_PREDICTION = False #If True, will generate a CSV to submit on AICrowd

PREDICTIONS_SAVE_DIR = 'predictions/'
MODELS_SAVE_DIR = 'model_save/'
MODEL_NAME = 'u-net' #Chose between cnn_handmade and u-net

TRAINING_SAMPLES = 10 #max 100

TRAINING_SIZE = 80 # Size of the training set in percentage, integer between 0 and 100, the remaining part is for testing
VALIDATION_SIZE = 0.20  # Size of the validation set, float between 0 and 1
SEED = 66478  # Set to None for random seed.
NUM_EPOCHS = 15

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
BATCH_SIZE = 16  # 64
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
IMAGE_SIZE = 400
if IMAGE_SIZE % IMG_PATCH_SIZE != 0 :
    print('Warning : the image size is not a multiple of the patch size')
    
#TODO move into a python file for Unet model
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

#TODO move into a python file for Unet model
def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = models.Model(inputs, outputs)
    return model

models_dict = {
    'u-net' : {
        'name' : 'u-net',
        'model' : UNet,
        'save_dir' : MODELS_SAVE_DIR + 'cnn_handmade/'
    }}

model_chosen = models_dict[MODEL_NAME]

data_dir = 'data/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/' 

def extract_data(filename, num_images):
    imgs = []
    for i in tqdm(range(1, num_images+1)):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
            
    print('Loaded {} training images'.format(len(imgs)))
    return np.asarray(imgs)

images = extract_data(train_data_filename, TRAINING_SAMPLES)
groundtruths = extract_data(train_labels_filename, TRAINING_SAMPLES)

print(images[0].shape)
images_mean = images#.mean(axis=3)
print(images_mean[0].shape)
print(groundtruths[0].shape)

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.keras import layers, models

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))

X_train, X_test, y_train, y_test = train_test_split(images_mean, groundtruths,\
                                                    train_size= TRAINING_SIZE/100, random_state=SEED)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

if RESTORE_MODEL:
    # It can be used to reconstruct the model identically.
    model = models.load_model(model_chosen['save_dir'])
else: 
    #This cell is required for me (Ben), otherwise I get "convolution algorithm not found"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("Set to true")
        print(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = model_chosen['model']()
    model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs = NUM_EPOCHS ,validation_split=VALIDATION_SIZE)
    
pred_train = model.predict(X_train)
print(pred_train)
print("Training error rate: {:.2f}%".format(error_rate(pred_train, y_train)))

pred_test = model.predict(X_test)
print("Test error rate: {:.2f}%".format(error_rate(pred_test, y_test)))

F1_score(y_test, pred_test)

print("Running prediction on training set")
prediction_training_dir = PREDICTIONS_SAVE_DIR + "predictions_training/"
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)
for i in range(1, TRAINING_SIZE + 1):
    pimg = get_prediction_with_groundtruth(model, train_data_filename, i)
    Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
    oimg = get_prediction_with_overlay(model, train_data_filename, i)
    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")  