import numpy as np
import matplotlib.pyplot as plt
from .image_processing import concatenate_images, prediction_to_rgb_image
from .constants import *


def general_check(predictions):
    print('Minimum and maximum prediction : ', predictions.max(), predictions.min())
    print('Road ratio : ', predictions.mean())


def visualize_random_predictions(x, y, predictions, size=4):
    fig, axes = plt.subplots(size, 4, figsize=(20, 5*size))
    for i, idx in enumerate(np.random.randint(len(x), size=size)):
        pred = predictions[idx]
        binarized_pred = (pred >= ROAD_THRESHOLD_PIXEL_PRED) * 1.0
        axes[i,0].imshow(x[idx])
        axes[i,1].imshow(prediction_to_rgb_image(pred))
        axes[i,2].imshow(prediction_to_rgb_image(binarized_pred))
        axes[i,3].imshow(prediction_to_rgb_image(y[idx]))
