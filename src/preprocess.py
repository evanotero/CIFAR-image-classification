import problem_unittests as tests
from sklearn import preprocessing
import numpy as np
import helper

cifar10_dataset_folder_path = '../cifar-10-batches-py'

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    newMax = 1
    newMin = 0

    return (x-newMin) *(float(newMax - newMin))/(255-0) + newMin

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    x_out = np.zeros((len(x), 10))

    for i, label in enumerate(x):
        onehot_enc = np.zeros(10)
        onehot_enc[label] = 1
        x_out[i, :] = onehot_enc

    return x_out

def test():
    tests.test_normalize(normalize)
    tests.test_one_hot_encode(one_hot_encode)

helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
