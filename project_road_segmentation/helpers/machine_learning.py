from .constants import *
from .file_manipulation import *
from sklearn.model_selection import train_test_split

def get_train_test(data_augmentation = False, transformations=None):
    """
    Load training images and split them into a training and testing sets.
    :param data_augmentation: set to `True` to use generated data
    :param transformation: `list` used to specify with generated data to use when 
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

