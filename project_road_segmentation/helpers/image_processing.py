import numpy as np
from .constants import *


def make_prediction(predicted_images, filename='submission.csv'):
    file = open(filename, "w")
    for image_id, pred in np.ndenumerate(predicted_images):
        patches_pred = make_patch_list(pred)
        for (x, y, p) in patches_pred:
            file.write("{:03}_{}_{},{}\n".format(image_id, x, y, p))
    file.close()


def make_patch_list(img_pred):
    """Transform pixel predictions into patch predictions. Format for csv is imgageid_x_y,pred"""
    image_width = img_pred.shape[0]
    image_height = img_pred.shape[1]
    if (image_width % patch_width != 0) or (image_height % patch_height != 0):
        raise ValueError("Image cannot be divided in patches of this size")
    patch_pred_list = []
    for i in range(0, image_height, patch_height):
        for j in range(0, image_width, patch_width):
            patch_pred = 1 if img_pred[j:j+patch_width, i:i+patch_height].mean() > ROAD_THRESHOLD_PATCH else 0
            patch_pred_list.append((j, i, patch_pred))
    return list_patches


def predict_patch(p):
    return 1 if p.mean() > ROAD_THRESHOLD_PATCH else 0


def concatenate_images(img1, img2):
    return np.concatenate((img1, img2), axis=1)


def prediction_to_rgb_image(prediction):
    return np.stack((prediction, prediction, prediction), axis=-1)
