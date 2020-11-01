"""Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
"""
import os
import pickle
import sys

# from __future__ import print_function
# import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

# Set the seed for hash based operations in python
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(111)
tf.random.set_seed(111)


def filter_base(x, y, base_class):
    keep = np.where(y == base_class)[0]
    y = y[keep]
    x = x[keep, :]
    print(keep.shape, y.shape, x.shape)

    return x, y


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# class specificaton
base_class = 4
target_class = 9

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#
# filter down to one class in training and test - SIMPLIFICATION OF PROBLEM
#
x_base_train, y_base_train = filter_base(x_train, y_train, base_class)
x_base_test, y_base_test = filter_base(x_test, y_test, base_class)

x_target_train, y_target_train = filter_base(x_train, y_train, target_class)
x_target_test, y_target_test = filter_base(x_test, y_test, target_class)

x_train = np.concatenate([x_base_train, x_target_train])
x_test = np.concatenate([x_base_test, x_target_test])


if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# combine y's into our nx2 matrix
y_train = np.zeros((x_train.shape[0], 2))
num_base = x_base_train.shape[0]
num_class = x_target_train.shape[0]
y_train[:num_base, 0] = 1
y_train[-num_class:, 1] = 1


y_test = np.zeros((x_test.shape[0], 2))
num_base = x_base_test.shape[0]
num_class = x_target_test.shape[0]
y_test[:num_base, 0] = 1
y_test[-num_class:, 1] = 1


# sys.exit()

act1 = 'tanh'
#ki = "he_normal"
ki = 'glorot_normal'
#ki = tf.keras.initializers.he_normal(seed=111)

model = Sequential()
model.add(
    Conv2D(
        16,
        kernel_size=(3, 3),
        activation=act1,
        kernel_initializer=ki,
        input_shape=input_shape,
    )
)
model.add(Conv2D(16, (3, 3), activation=act1, kernel_initializer=ki))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Conv2D(8, (3, 3), activation=act1, kernel_initializer=ki))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), activation=act1, kernel_initializer=ki))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation=act1, kernel_initializer=ki))
model.add(Dense(2, activation="sigmoid"))


print(model.summary())
# sys.exit()


model.compile(
    # loss=keras.losses.categorical_crossentropy,
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


model_path = "./models/"
try:
    os.mkdir(model_path)
except OSError as error:
    print("Skipping creation of models directory - ", error)

model.save("models/00_classifier.h5")

#
# dump predictions out to pkl file!
#
# pred = model.predict(x_test)
# with open("x_test_pred_classes.pkl", "wb") as f:
#    pickle.dump(pred, f)
