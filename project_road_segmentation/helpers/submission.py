import os
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from .constants import *
from .file_manipulation import *


def predict_submissions(model):
    """
    Given a model, runs the whole pipeline to create a file containing the label of each patch,
    ready for the AIcrowd submission.
    Saves the csv in the folder `SUBMISSIONS_DIR`
    Writes the predicted masks in the folder `PREDICTIONS_SAVE_DIR`
    """
    #Load 608x608 test images 
    images608 = load_test_images()
    
    #Split each of them into 4 400x400 images
    images400 = split_608_to_400(images608)
    
    #Predict their mask
    print("Predicting test images...")
    masks400 = model.predict(images400).squeeze()
    
    #Merge the 4 400x400 masks into one 608x608 by averaging the overlapping parts
    masks608 = merge_masks400(masks400)
    
    # binarize predictions into 0's and 1's
    masks608 = np.asarray([(mask >= ROAD_THRESHOLD_PIXEL_PRED) * 1.0 for mask in masks608])
 
    #Convert mask to patch labels and write them into the file submission.csv
    if not os.path.isdir(SUBMISSIONS_DIR):
        os.mkdir(SUBMISSIONS_DIR)
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    make_prediction(masks608, filename=f"{SUBMISSIONS_DIR}submission_{now}.csv")
    
    # Write the masks to folder
    write_predictions(masks608)


def split_608_to_400(images608):
    """
    Split a 608x608 image into 4 400x400 images in the following order:
    | 1 | 2 |
    | 3 | 4 |
    :param images608: list of 608x608 images
    """
    thres1 = TRAINING_IMG_SIZE #End of the top-left masks
    thres2 = TEST_IMG_SIZE - TRAINING_IMG_SIZE #Start of the bottom right masks
    
    images400 = []
    for img in images608:
        images400.append(img[:thres1,:thres1])
        images400.append(img[:thres1,thres2:])
        images400.append(img[thres2:,:thres1])
        images400.append(img[thres2:,thres2:])
    return np.asarray(images400)


def merge_masks400(masks400):
    """
    Merge 4 400x400 masks where images of the list `[img1,img2,img3,img4, ...]` corresponds to the following order:
    | img1 | img2 |
    | img3 | img4 | into a 608x608 mask. Overlapping parts are averaged among the predictions.
    :params masks400: list of 400x400 masks, where each consecutive 4 masks corresponds to a 608x608 mask
    """
    if len(masks400) % 4 != 0:
        raise ValueError("Number of 400x400 predictions is not a mulitple of 4.")

    thres1 = TRAINING_IMG_SIZE #End of the top-left masks
    thres2 = TEST_IMG_SIZE - TRAINING_IMG_SIZE #Start of the bottom-right masks
    thres3 = thres1 - thres2
    
    list_masks608 = []
    for idx in range(0, len(masks400), 4):
        mask608 = np.empty((TEST_IMG_SIZE, TEST_IMG_SIZE), dtype='float32') #initialize a 608x608 mask
        img1, img2, img3, img4 = [masks400[idx + i] for i in range(4)]
        
        #Copy parts that belong to a single mask into the final mask, 
        #These parts correspond to corners of the final mask
        mask608[:thres2, :thres2] = img1[:thres2, :thres2]
        mask608[:thres2, thres1:] = img2[:thres2, thres3:]
        mask608[thres1:, :thres2] = img3[thres3:, :thres2]
        mask608[thres1:, thres1:] = img4[thres3:, thres3:]
        
        #Average parts that belong to multiple masks
        mask608[:thres2, thres2:thres1] = (img1[:thres2, thres2:] + img2[:thres2, :thres3]) / 2
        mask608[thres2:thres1, :thres2] = (img1[thres2:, :thres2] + img3[:thres3, :thres2]) / 2
        mask608[thres2:thres1, thres2:thres1] = (img1[thres2:, thres2:] + img2[thres2:, :thres3] +
                                                img3[:thres3,thres2:] + img4[:thres3, :thres3]) / 4
        mask608[thres2:thres1, thres1:] = (img2[thres2:, thres3:] + img4[:thres3, thres3:]) / 2
        mask608[thres1:, thres2:thres1] = (img3[thres3:, thres2:] + img4[thres3:, :thres3]) / 2
        list_masks608.append(mask608)
    return list_masks608


def make_prediction(predicted_images, filename='submission.csv'):
    """
    Create the submission csv containing the label of all the patches 
    of each images in the test folder.
    :param predicted_images: list of 608x608 masks
    """
    print(f"Creating submission csv at location {filename}")
    file = open(filename, "w")
    file.write('id,prediction\n')
    for image_id, pred in enumerate(predicted_images):
        patches_pred = make_patch_list(pred)
        for (x, y, p) in patches_pred:
            file.write(f"{image_id + 1:03}_{y}_{x},{p}\n")
    file.close()


def make_patch_list(img_pred):
    """Transform pixel predictions into patch predictions. Format for csv is imgageid_x_y,pred"""
    image_width = img_pred.shape[0]
    image_height = img_pred.shape[1]
    if (image_width % PATCH_SIZE != 0) or (image_height % PATCH_SIZE != 0):
        raise ValueError("Image cannot be divided in patches of this size")
    patch_pred_list = []
    for i in range(0, image_height, PATCH_SIZE):
        for j in range(0, image_width, PATCH_SIZE):
            patch_pred = 1 if img_pred[j:j+PATCH_SIZE, i:i+PATCH_SIZE].mean() > ROAD_THRESHOLD_PATCH else 0
            patch_pred_list.append((j, i, patch_pred))
    return patch_pred_list

    
def write_predictions(predictions, folder=PREDICTIONS_SAVE_DIR):
    """Save the predicted masks in the specified folder."""
    if not os.path.isdir(folder):
        os.mkdir(folder)
    print(f"Writing predictions in folder {folder}")
    for i, pred in enumerate(predictions):
         mpimg.imsave(f"{folder}test_{i+1}.png", pred, cmap='gray')


