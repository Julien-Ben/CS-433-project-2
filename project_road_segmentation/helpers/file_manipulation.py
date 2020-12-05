import os
import re
import numpy as np
import matplotlib.image as mpimg
from PIL import Image, ImageOps
from tqdm import tqdm
from .constants import *
from .image_processing import predict_patch


def masks_to_submission(submission_filename, image_dir):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as file:
        file.write('id,prediction\n')
        for img_name in tqdm(os.listdir(image_dir)):
            file.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_dir + img_name))


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    for j in range(0, im.shape[1], PATCH_SIZE):
        for i in range(0, im.shape[0], PATCH_SIZE):
            patch = im[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            label = predict_patch(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def load_images(path, num_images):
    images = []
    for i in tqdm(range(num_images), desc="Loading " + path):
        image_id = "satImage_%.3d" % (i+1)
        image_filename = path + image_id + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            images.append(img)
        else:
            raise ValueError('File ' + image_filename + ' does not exist')

    return np.asarray(images)


def load_test_images(path=TEST_IMAGES_DIR, num_images=50):
    images = []
    for i in tqdm(range(num_images)):
        image_id = "test_{}/test_{}".format(i+1, i+1)
        image_filename = path + image_id + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            images.append(img)
        else:
            raise ValueError('File ' + image_filename + ' does not exist')

    return np.asarray(images)


def load_features(num_images):
    return load_images(TRAIN_IMAGES_DIR, num_images)


def load_labels(num_images):
    """
    Loads all labels.
    Since the pixels are not perfectly black and white we use an artificial threshold
    """
    gt = load_images(TRAIN_LABELS_DIR, num_images)
    return 1.0*(gt > ROAD_THRESHOLD_PIXEL)

def load_folder(path, grayscale=False):
    """
    Load every image in the fodler at path with name format 'satImage'
    """
    imgs = []
    image_names = sorted([path + image for image in os.listdir(path) if 'satImage' in image])
    for image_name in tqdm(image_names, desc="Loading " + path):
        img = Image.open(image_name).convert('RGB')
        if grayscale:
            img = ImageOps.grayscale(img)
        img = np.array(img)
        imgs.append(img)
    return imgs

def load_generated_data(transformations=None):
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
    
    images = []
    groundtruths = []
    for folder in folders_to_load:
        image_path = GENERATION_DIR + folder + '/images/'
        gt_path = GENERATION_DIR + folder + '/groundtruth/'
        images.append(load_folder(image_path))
        groundtruths.append(load_folder(gt_path, grayscale=True))
    return np.concatenate(images,axis=0), np.concatenate(groundtruths, axis=0)
    
