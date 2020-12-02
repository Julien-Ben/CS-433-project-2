import numpy as np
import matplotlib.pyplot as plt
from .image_processing import concatenate_images, prediction_to_rgb_image
from .constants import *


def general_check(predictions):
    print('Minimum and maximum prediction : ', predictions.max(), predictions.min())
    print('Road ratio : ', predictions.mean())


def visualize_random_predictions(X_train, y_train, predictions, size=4):
    for i in np.random.randint(int(TRAINING_SAMPLES * TRAINING_SIZE), size=size):
        pred = predictions[i].copy()
        pred = (pred >= ROAD_THRESHOLD_PIXEL_PRED) * 1.0
        plt.figure()
        img1 = concatenate_images(X_train[i], prediction_to_rgb_image(predictions[i]))
        img2 = concatenate_images(img1, prediction_to_rgb_image(pred))
        img3 = concatenate_images(img2, prediction_to_rgb_image(y_train[i]))
        plt.imshow(img3)
