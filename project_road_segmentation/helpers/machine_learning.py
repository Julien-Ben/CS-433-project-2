from .constants import *
from .file_manipulation import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.preprocessing.image import apply_affine_transform, apply_brightness_shift
import pandas as pd
import numpy as np


def get_train_test(data_augmentation=False, transformations=None):
    """
    Load training images and split them into a training and testing sets.
    :param data_augmentation: set to `True` to use generated data
    :param transformations: `list` used to specify with generated data to use when
    data_augmentation is `True`. Leave to `None` to load everything. 
    Current possible values: `['mix', 'flip', 'shift', 'rotation']`
    """
    images = load_features(TRAINING_SAMPLES)
    groundtruths = load_labels(TRAINING_SAMPLES)
    if data_augmentation:
        images_gen, groundtruths_gen = load_generated_data(transformations)
        images = np.vstack([images, images_gen])
        groundtruths = np.vstack([groundtruths, groundtruths_gen])
    print('Training features shape : ', images.shape)
    print('Training labels shape : ', groundtruths.shape)
    X_train, X_test, y_train, y_test = train_test_split(images, groundtruths,
                                                        train_size=TRAINING_SIZE, random_state=SEED)
    return X_train, X_test, y_train, y_test


def get_train_test_feature_augmentation(rotations=0, thetas=[], feature_augmentation=0, brightnesses=[]):
    """
    Load training images and split them into a training and testing sets.
    """
    if not (len(brightnesses) == feature_augmentation and len(thetas) == rotations):
        raise ValueError('Wrong arguments')
    images = load_features(TRAINING_SAMPLES)
    groundtruths = load_labels(TRAINING_SAMPLES)

    augmented_features = np.empty(shape=(0, TRAINING_IMG_SIZE, TRAINING_IMG_SIZE, 3))
    augmented_labels = np.empty(shape=(0, TRAINING_IMG_SIZE, TRAINING_IMG_SIZE))
    for i in range(rotations):
        tmp_f = augment_data(images, thetas[i], True)
        tmp_l = augment_data(groundtruths, thetas[i], False)
        augmented_features = np.concatenate((augmented_features, tmp_f), axis=0)
        augmented_labels = np.concatenate((augmented_labels, tmp_l), axis=0)
    images = np.concatenate((images, augmented_features), axis=0)
    groundtruths = np.concatenate((groundtruths, augmented_labels), axis=0)

    print('Training features shape : ', images.shape)
    print('Training labels shape : ', groundtruths.shape)
    X_train, X_test, y_train, y_test = train_test_split(images, groundtruths,
                                                        train_size=TRAINING_SIZE, random_state=SEED)

    augmented = np.empty(shape=(len(X_train), TRAINING_IMG_SIZE, TRAINING_IMG_SIZE, 0))
    for i in range(feature_augmentation):
        tmp = augment_features(images, brightnesses[i])
        augmented = np.concatenate((augmented, tmp), axis=3)
    X_train = np.concatenate((X_train, augmented), axis=3)

    return X_train, X_test, y_train, y_test


def get_train_test_manual_split(transformations=None):
    """
    Load training images and split them into a training and testing sets.
    :param transformations: `list` used to specify with generated data to use when
    data_augmentation is `True`. Leave to `None` to load everything.
    Current possible values: `['mix', 'flip', 'shift', 'rotation']`
    """
    images = load_features(TRAINING_SAMPLES)
    groundtruths = load_labels(TRAINING_SAMPLES)
    images_gen, groundtruths_gen = load_generated_data(transformations)
    X_train = images
    y_train = groundtruths
    half = int(len(images)/2)
    X_test = images_gen[:half]
    y_test = groundtruths_gen[:half]
    validation_set = (images_gen[half:], groundtruths_gen[half:])
    return X_train, X_test, y_train, y_test, validation_set


def compute_metrics(y_true, y_pred):
    """
    Compute pixel-wise metrics on the predictions where arguments are 1d vectors.
    :params y_test: 1d vector with the true labels
    :params y_pred: 1d vectore with the predictions
    """
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    col = ['f1', 'acc', 'precision', 'recall']
    data = [[f1, acc, precision, recall]]
    return pd.DataFrame(data, columns=col, index=["metrics"])


def compute_entire_images_metrics(y_true, y_pred):
    """
    Compute pixel-wise metrics where arguments are lists of masks
    """
    y_pred = np.array((y_pred >= ROAD_THRESHOLD_PIXEL_PRED) * 1.0, dtype=int).ravel()
    y_true = np.array(y_true.ravel(), dtype=int)
    return compute_metrics(y_true, y_pred)


def augment_features(images, new_brightness):
    augmented = np.empty(shape=(0, TRAINING_IMG_SIZE, TRAINING_IMG_SIZE, 3))
    for im in images:
        tmp = apply_brightness_shift(im, new_brightness)/255
        augmented = np.r_[augmented, tmp[None, :, :, :]]
    return augmented


def augment_data(images, theta, is_RGB):
    shape = (0, TRAINING_IMG_SIZE, TRAINING_IMG_SIZE, 3) if is_RGB else (0, TRAINING_IMG_SIZE, TRAINING_IMG_SIZE)
    augmented = np.empty(shape=shape)
    for im in images:
        tmp = apply_affine_transform(im, theta=theta, fill_mode='reflect')
        augmented = np.r_[augmented, tmp[None, :, :, :]]
    return augmented
