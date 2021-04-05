import cv2
import pickle
import numpy as np

train_images = pickle.load(open("images_set.pickle", "rb"))

for x in range(0, 50):
    cv2.imshow("snip_threshed", train_images[x])
    cv2.waitKey(0)
