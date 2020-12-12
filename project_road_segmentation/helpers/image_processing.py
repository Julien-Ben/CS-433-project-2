import numpy as np
from .constants import *
from tqdm import tqdm
import os
from PIL import Image


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

def apply_masks_on_test(num_images=50, opacity=100):
    for i in tqdm(range(num_images), desc="Loading"):
        test_image_path = TEST_IMAGES_DIR + 'test_{}/test_{}.png'.format(i + 1, i + 1)
        mask_path = PREDICTIONS_SAVE_DIR + "test_{}.png".format(i + 1)
        if os.path.isfile(test_image_path) and os.path.isfile(mask_path):
            img = Image.open(test_image_path)
            mask = Image.open(mask_path)
            masked = mask_image(img, mask, opacity)
            masked_path = PREDICTIONS_SAVE_DIR + 'test_{}_with_mask.png'.format(i + 1)
            masked.save(masked_path)
        else:
            raise ValueError('Files {} or {} '.format(img, mask) + ' do not exist')


def mask_image(img, mask, opacity=100):
    mask = mask.convert('RGBA')
    pixels = mask.getdata()

    new_pixels = []
    for px in pixels:
        if px[0] == 255 and px[1] == 255 and px[2] == 255:
            new_pixels.append((255, 0, 0, opacity))
        else:
            new_pixels.append((0, 0, 0, 0))

    mask.putdata(new_pixels)
    img.paste(mask, None, mask)
    return img
