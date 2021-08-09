import os
import gzip
import numpy as np
import cv2
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def resize_mnist(img_array, IMG_SIZE1, IMG_SIZE2):
    tmp = np.empty((img_array.shape[0], IMG_SIZE1, IMG_SIZE1))

    for i in range(len(img_array)):
        img = img_array[i].reshape(IMG_SIZE2, IMG_SIZE2).astype('uint8')
        img = cv2.resize(img, (IMG_SIZE1, IMG_SIZE1))
        img = img.astype('float32')/255
        tmp[i] = img
        
    return tmp

