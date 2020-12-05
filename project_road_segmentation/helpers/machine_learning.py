from .constants import *
from .file_manipulation import *
from sklearn.model_selection import train_test_split


def get_train_test():
    images = load_features(TRAINING_SAMPLES)
    groundtruths = load_labels(TRAINING_SAMPLES)
    print('Training features shape : ', images.shape)
    print('Training labels shape : ', groundtruths.shape)
    X_train, X_test, y_train, y_test = train_test_split(images, groundtruths,
                                                        train_size=TRAINING_SIZE, random_state=SEED)
    return X_train, X_test, y_train, y_test

