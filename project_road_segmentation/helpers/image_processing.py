import os
import re
import numpy as np
import matplotlib.image as mpimg
from PIL import Image, ImageOps
from tqdm import tqdm
from .constants import *
from .image_processing import predict_patch

def load_images(images, path, num_images):
    for i in tqdm(range(num_images), desc="Loading " + path):
        image_id = "satImage_%.3d" % (i+1)
        image_filename = path + image_id + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            images.append(img)
        else:
            raise ValueError('File ' + image_filename + ' does not exist')

    return np.asarray(images)


def load_test_images(images, path=TEST_IMAGES_DIR, num_images=50):
    for i in tqdm(range(num_images), desc="Loading " + path):
        image_id = "test_{}/test_{}".format(i+1, i+1)
        image_filename = path + image_id + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            images.append(img)
        else:
            raise ValueError('File ' + image_filename + ' does not exist')

    return np.asarray(images)


def load_features(images, num_images):
    return load_images(images, TRAIN_IMAGES_DIR, num_images)


def load_labels(gt, num_images):
    """
    Loads all labels.
    Since the pixels are not perfectly black and white we use an artificial threshold
    """
    load_images(gt, TRAIN_LABELS_DIR, num_images)
    return list(1.0*(np.array(gt) > ROAD_THRESHOLD_PIXEL))

def load_folder(imgs, path, grayscale=False):
    """
    Load every image in the folder at path with name format 'satImage'
    """
    image_names = sorted([path + image for image in os.listdir(path) if 'satImage' in image])
    for image_name in tqdm(image_names, desc="Loading " + path):
        img = Image.open(image_name).convert('RGB') #Convert rgba to rgb
        if grayscale:
            img = ImageOps.grayscale(img) #Convert to grayscale
        img = np.array(img) / 255 #Scale value between 0 and 1
        imgs.append(img)
    # return imgs

def load_generated_data(images, groundtruths, transformations=None):
    """
    :param transformations: list of transformation folders to load, if None or empty loads everything, 
    current possible values: ['mix', 'flip', 'shift', 'rotation']
    """
    # List all possible folders we can load
    # Condition using isdir to exclude files like .DS_Store etc.
    folders_to_load = [folder for folder in os.listdir(GENERATION_DIR) if os.path.isdir(GENERATION_DIR + folder)]

    # If specific folders are specified
    if transformations != None and len(transformations) > 0:
        folders_to_load = [folder for folder in transformations if folder in folders_to_load]
    
    # images = []
    # groundtruths = []
    for folder in folders_to_load:
        image_path = GENERATION_DIR + folder + '/images/'
        gt_path = GENERATION_DIR + folder + '/groundtruth/'
        load_folder(images, image_path)
        load_folder(groundtruths, gt_path, grayscale=True)
        
    return images, groundtruths


