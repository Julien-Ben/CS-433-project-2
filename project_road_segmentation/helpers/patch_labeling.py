from .constants import *
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

#Extract blocks of dim blocksize * blocksize from an image a
def extract_blocks(a, blocksize, keep_as_view=False):
    #https://stackoverflow.com/a/31530106
    M,N = a.shape
    b0 = blocksize
    b1 = blocksize
    if keep_as_view==0:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2).reshape(-1,b0,b1)
    else:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2)

#Apply extract_blocks to an array of labels
def label_to_patches(labels):
    patches = []
    for i in range(len(labels)):
        patches.append(extract_blocks(labels[i], PATCH_SIZE))
    return np.array(patches)

def patches_to_predictions(patches):
    patches_labeled = []
    for index_i,i in enumerate(patches):
        patches_labeled.append([])
        for index_j, j in enumerate(i):
            patches_labeled[index_i].append(predict_patch(j))
    label_array = np.array(patches_labeled)
    #return np.clip(label_array.reshape(TRAINING_IMG_SIZE,PATCH_SIZE,PATCH_SIZE).astype(np.float32), 0, 1)
    return label_array.reshape(TRAINING_IMG_SIZE, PATCH_SIZE, PATCH_SIZE)

def label_to_patch_prediction(labels):
    return patch_to_predictions(label_to_patches(labels))

def visualize_patches(original, patch_array, number=5):
    resized_patch = resize_images(patch_array, IMG_SIZE)
    for i in np.random.randint(len(resized_patch), size=number):
        plt.figure()
        plt.imshow(concatenate_images(original[i], resized_patch[i]))

def resize_images(img_array, newsize):
    return resize(img_array, (len(img_array), newsize, newsize))