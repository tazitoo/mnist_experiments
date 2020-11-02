#!/usr/bin/env python
import gzip
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix as cm
# based on 
# https://gist.githubusercontent.com/ischlag/41d15424e7989b936c1609b53edd1390/raw/5ed7aca47bcca30b3df1c3bfd0f027e6bcdb430c/mnist-to-jpg.py

WORK_DIRECTORY = '../qmnist/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

# class specificaton
base_class = 4
target_class = 9


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

def filter_base(x, y, base_class):
    keep = np.where(y == base_class)[0]
    y = y[keep]
    x = x[keep, :]
    print(keep.shape, y.shape, x.shape)

    return x, y

# images
img_file = WORK_DIRECTORY + 'qmnist-test-images-idx3-ubyte.gz'
qtest = extract_data(img_file, 60000)
qtest /= 255

label_file = WORK_DIRECTORY + 'qmnist-test-labels-idx1-ubyte.gz'
qtest_labels = extract_labels(label_file, 60000)

x_base_test, y_base_test = filter_base(qtest, qtest_labels, base_class)
x_target_test, y_target_test = filter_base(qtest, qtest_labels, target_class)

# load model
classifier = load_model('models/a_classifier.h5', compile='False')

base_pred = classifier.predict(x_base_test)
y_base_pred = base_pred[:, 0] > 0.5

base_cm = cm(y_base_test, y_base_pred)
