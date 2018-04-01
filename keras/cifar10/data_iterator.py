import numpy as np


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
    # print('filesnames %s' % filename)
    # print('metadata %s' % metadata)

    # we load the image and reshape it to a vector
    x = read_image(filename)
    # convert the class number to one hot
    y = one_hot(int(metadata['label_index']))
    return x, y
