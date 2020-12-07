import numpy as np
from .constants import *


def predict_patch(p):
    return 1 if p.mean() > ROAD_THRESHOLD_PATCH else 0


def concatenate_images(img1, img2):
    return np.concatenate((img1, img2), axis=1)


def prediction_to_rgb_image(prediction):
    return np.stack((prediction, prediction, prediction), axis=-1)

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg
