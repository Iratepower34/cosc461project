import pickle

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import datasets

tensorBoard = TensorBoard(log_dir="logs/model")
train_images = pickle.load(open("images_set.pickle", "rb"))
train_labels = pickle.load(open("labels_set.pickle", "rb"))

train_images = train_images / 255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = train_images.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(.2))

model.add(Dense(14))
model.add(Activation('softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, epochs=3, callbacks=[tensorBoard])
model.save("models/test.model")
