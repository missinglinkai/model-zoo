# -- coding: utf8 --
import random

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

def read_image(filename):
    import matplotlib.image as mpimg

    img = mpimg.imread(filename)
    # reshape it from (32,32,3) to a vector of 3072 numbers
    new_img = img # .reshape((-1), order='F')

    return new_img


def one_hot(i):
    a = np.zeros(10, 'uint8')
    a[i] = 1

    return a


def process_file_and_metadata(filename, metadata):
    """
    This function will be called for each data point.
    In our case - this is coming from CIFAR10 and will include 1 file in the file_names and the respected
    metadata which includes a property called 'label_index;
    :rtype: a tuple with inputs of the model. a numpy array with the image and a one hot vector for the class
    :param filename: the filename for the datapoint
    :param metadata: the metadata for the filenames
    """
    # if random.randint(0, 100) % 10:
    #     print('filesnames %s' % filename)
    #     print('metadata %s' % metadata)

    # we load the image and reshape it to a vector
    x = read_image(filename)
    x = datagen.random_transform(x)
    # convert the class number to one hot
    y = one_hot(int(metadata['label_index']))
    return x, y
