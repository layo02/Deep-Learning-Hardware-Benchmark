import os
import gzip
import numpy as np
import cv2

def load_mnist(path, kind, **kwargs):
    """
    Load the MNIST dataset from a defined directory.
    Args:
        path: the directory to the MNIST dataset, for example 'data/mnist'
        kind: type of data that is either train or test.
    Returns:
        images, labels: the arrays of MNIST dataset.
    
    """
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 784)

    return images, labels


def resize_mnist(img_array, OR_IMG_SIZE1, OR_IMG_SIZE2, NEW_IMG_SIZE1, NEW_IMG_SIZE2, *args):
    """
    Return the resized array of MNIST images with a new resolution that is compatible with several CNNs such as VGG19, etc.
    
    Args:
        img_array: an array of MNIST images.
        OR_IMG_SIZE1, OR_IMG_SIZE2: height and width of each MNIST image, respectively.
        NEW_IMG_SIZE1, NEW_IMG_SIZE2: new height and width of each MNIST image, respectively.
    Returns:
        tmp: the resized array of MNIST images with a new resolution.
    """
    ##
    # Create an empty array to store the resized images:
    #
    tmp = np.empty((img_array.shape[0], NEW_IMG_SIZE1, NEW_IMG_SIZE2))
    
    ##
    # Resize each image and store it to the empty array:
    #
    for i in range(len(img_array)):
        img = img_array[i].reshape(OR_IMG_SIZE1, OR_IMG_SIZE2).astype('uint8')
        img = cv2.resize(img, (NEW_IMG_SIZE1, NEW_IMG_SIZE2))
        img = img.astype('float32')/255
        tmp[i] = img
        
    return tmp
