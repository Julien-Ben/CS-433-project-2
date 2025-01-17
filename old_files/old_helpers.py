import gzip
import os
from tqdm import tqdm
import matplotlib.image as mpimg
from PIL import Image

import numpy as np
import tensorflow as tf

from constants import *


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
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
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in tqdm(range(1, num_images + 1)):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    print('Loaded {} groudtruth images'.format(len(gt_imgs)))
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg



def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx=0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(input_tensor=V)
    V = V - min_value
    max_value = tf.reduce_max(input_tensor=V)
    V = V / (max_value*PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(a=V, perm=(2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(a=V, perm=(2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

# Get prediction for given input image 
def get_prediction(model, img):
    data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    output_prediction = model.predict(data)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    return img_prediction

# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(model, filename, image_idx):
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    img_prediction = get_prediction(model, img)
    cimg = concatenate_images(img, img_prediction)
    return cimg

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(model, filename, image_idx):
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    img_prediction = get_prediction(model, img)
    oimg = make_img_overlay(img, img_prediction)
    return oimg

def one_hot_to_scalar(one_hot_labels):
    """
    Convert one-hot encoded labels to a one-dimensional array
    Ex: [[0,1], [1,0]] becomes [1,0]
    """
    return np.argmax(one_hot_labels, axis=1)

def compute_confusion_matrix(y_truth, y_pred):
    """
    Compute the confusion matrix of the predicted labels.
    Arguments are one-hot encoded
    """
    y_truth_vec = one_hot_to_scalar(y_truth)
    y_pred_vec = one_hot_to_scalar(y_pred)
    TP = np.count_nonzero(y_pred_vec * y_truth_vec)
    TN = np.count_nonzero((y_pred_vec - 1) * (y_truth_vec - 1))
    FP = np.count_nonzero(y_pred_vec * (y_truth_vec - 1))
    FN = np.count_nonzero((y_pred_vec - 1) * y_truth_vec)
    return TP, TN, FP, FN

def F1_score(y_truth, y_pred):
    """
    Compute the F1 score. Arguments are one-hot encoded
    """
    TP, TN, FP, FN = compute_confusion_matrix(y_truth, y_pred)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2 * precision * recall / (precision + recall)

def prediction_to_img(pred):
    """
    Convert a prediction array (containing only 0s and 1s) to 
    a 3-channel image (containing only 0s and 255s) that can be displayed.
    """
    pred3c = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    pred8 = img_float_to_uint8(pred)          
    pred3c[:, :, 0] = pred8
    pred3c[:, :, 1] = pred8
    pred3c[:, :, 2] = pred8
    return pred3c

def predict_test_masks(model, test_dir, prediction_test_dir):   
    """
    Predict the masks associated with the test images. The masks
    are saved in the folder prediction_test_dir.
    """ 
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)
    for img_dir in tqdm(os.listdir(test_dir)):
         if "test_" in img_dir:
            image_filename = test_dir + img_dir+'/'+img_dir + ".png"
            img = mpimg.imread(image_filename)
            mask = prediction_to_img(get_prediction(model, img))
            Image.fromarray(mask).save(prediction_test_dir + img_dir + ".png")

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