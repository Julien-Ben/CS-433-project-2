from .constants import *
from .file_manipulation import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np

def get_train_test(images, groundtruths, data_augmentation=False, transformations=None):
    """
    Load training images and split them into a training and testing sets.
    :param data_augmentation: set to `True` to use generated data
    :param transformation: `list` used to specify with generated data to use when 
    data_augmentation is `True`. Leave to `None` to load everything. 
    Current possible values: `['mix', 'flip', 'shift', 'rotation']`
    """
    np.random.seed(SEED)
    load_features(images, TRAINING_SAMPLES)
    load_labels(groundtruths, TRAINING_SAMPLES)
    if data_augmentation:
        load_generated_data(images, groundtruths, transformations)
        # images = np.vstack([images, images_gen])
        # groundtruths = np.vstack([groundtruths, groundtruths_gen])
    print('Training features shape : ', np.array(images).shape)
    print('Training labels shape : ', np.array(groundtruths).shape)
    # X_train, X_test, y_train, y_test = train_test_split(images, groundtruths,
    #                                                     train_size=TRAINING_SIZE, random_state=SEED)
    # return X_train, X_test, y_train, y_test
    idx = np.arange(len(images))
    np.random.shuffle(idx)
    split = int(TRAINING_SIZE * len(images))
    train_idx, test_idx = idx[:split], idx[split:] 
    return train_idx, test_idx

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
    return compute_metrics(y_true , y_pred)