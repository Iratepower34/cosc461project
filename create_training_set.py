import numpy as np
import tensorflow as tf
import os
import cv2
import random
import pickle
from tensorflow.keras import datasets

def resize_image(img, size=(64,64)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus", "minus", "times", "divide"]
training_set = []

for i, label in enumerate(labels):
    path = os.path.join("training_set", label)
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

        snip_thresh = cv2.dilate(img, np.ones((1, 1),np.uint8))
        snip_thresh = cv2.bitwise_not(snip_thresh)
        snip_thresh = cv2.copyMakeBorder(snip_thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, 0)

        result = resize_image(snip_thresh, (64,64))
        training_set.append((result, i))

random.shuffle(training_set)

images_set = []
labels_set = []

for image, label in training_set:
    images_set.append(image)
    labels_set.append(label)

images_set = np.array(images_set).reshape(-1, 64, 64, 1)
labels_set = np.array(labels_set)

pickle_out = open("images_set.pickle", "wb")
pickle.dump(images_set, pickle_out)
pickle_out.close()

pickle_out = open("labels_set.pickle", "wb")
pickle.dump(labels_set, pickle_out)
pickle_out.close()
