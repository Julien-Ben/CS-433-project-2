


from skimage.transform import resize
bottle_resized = resize(test_predictions, (100, 400, 400))
plt.imshow(bottle_resized[3])
plt.figure()
plt.imshow(test_predictions[3])