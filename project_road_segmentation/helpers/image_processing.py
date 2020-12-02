import numpy as np
from .constants import *


def extract_patches(im, patch_width, patch_height):
    """Returns a list of w*h patches from the image"""
    image_width = im.shape[0]
    image_height = im.shape[1]
    if (image_width % patch_width != 0) or (image_height % patch_height != 0):
        raise ValueError("Image cannot be divided in patches of this size")
    list_patches = []
    is_2d = len(im.shape) < 3
    for i in range(0, image_height, patch_height):
        for j in range(0, image_width, patch_width):
            if is_2d:
                im_patch = im[j:j+patch_width, i:i+patch_height]
            else:
                im_patch = im[j:j+patch_width, i:i+patch_height, :]
            list_patches.append(im_patch)
    return list_patches


def predict_patch(p):
    return 1 if p.mean() > ROAD_THRESHOLD_PATCH else 0


def concatenate_images(img1, img2):
    return np.concatenate((img1, img2), axis=1)


def prediction_to_rgb_image(prediction):
    return np.stack((prediction, prediction, prediction), axis=-1)
