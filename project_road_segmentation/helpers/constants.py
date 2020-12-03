# Section 1 : Not parameters (don't try and tune these)
ROAD_THRESHOLD_PATCH = .25
# Set image patch size in pixels
# PATCH_SIZE should be a multiple of 4 and divide the images' size
PATCH_SIZE = 16
PIXEL_DEPTH = 255

# Section 2 : Hyperparameters
PREDICTIONS_SAVE_DIR = 'predictions/'
MODELS_SAVE_DIR = 'model_save/'
TRAIN_IMAGES_DIR = 'data/training/images/'
TEST_IMAGES_DIR = 'data/test_set_images/'
TRAIN_LABELS_DIR = 'data/training/groundtruth/'

RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
SAVE_MODEL = False
GENERATE_PREDICTION = False  # If True, will generate a CSV to submit on AICrowd

MODEL_NAME = 'unet_1'  # For now, cnn, unet-1, unet-2
SAVE_DIR = MODELS_SAVE_DIR + MODEL_NAME + '/'

SEED = 66478  # Set to None for random seed.
TRAINING_SAMPLES = 100  # Max is 100
VALIDATION_SIZE = .2  # Remaining part is for training and testing
TRAINING_SIZE = .8  # Remaining part is for testing
NUM_EPOCHS = 50

ROAD_THRESHOLD_PIXEL = 0.5
ROAD_THRESHOLD_PIXEL_PRED = 0.5
