from .constants import *
from .file_manipulation import *
from sklearn.model_selection import train_test_split

def get_train_test(data_augmentation = False, transformations=None):
    images = load_features(TRAINING_SAMPLES)
    groundtruths = load_labels(TRAINING_SAMPLES)
    print('Training features shape : ', images.shape)
    print('Training labels shape : ', groundtruths.shape)
    if data_augmentation:
        images_gen, groundtruths_gen = load_generated_data(transformations)
        images = np.concat([images, images_gen], axis=0)
        groundtruths = np.concat([groundtruths, groundtruths_gen], axis=0)
        print('Training features shape with data augmentation : ', images.shape)
        print('Training labels shape with data augmentation : ', groundtruths.shape)
    X_train, X_test, y_train, y_test = train_test_split(images, groundtruths,
                                                        train_size=TRAINING_SIZE, random_state=SEED)
    return X_train, X_test, y_train, y_test

