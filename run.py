import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus", "minus", "times", "divide"]

model=tf.keras.models.load_model("test.model")
image = cv2.imread("testing-pics/equalsTest.png", cv2.IMREAD_GRAYSCALE)
image = cv2.bitwise_not(image)
array = image.reshape(1, 64, 64, 1) / 255

label_indexes = list(map(np.argmax, model.predict([array])))
print([labels[index] for index in label_indexes])
