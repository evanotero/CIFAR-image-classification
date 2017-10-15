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
    x_out = []
    for image in x:
        min_max_scaler = preprocessing.MaxAbsScaler()

        dim_1 = min_max_scaler.fit_transform(image[:, :, 0])
        dim_2 = min_max_scaler.fit_transform(image[:, :, 1])
        dim_3 = min_max_scaler.fit_transform(image[:, :, 2])

        new_image = np.zeros((32, 32, 3))
        new_image[:, :, 0] = dim_1
        new_image[:, :, 1] = dim_2
        new_image[:, :, 2] = dim_3

        x_out.append(new_image)

    return np.array(x_out)

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