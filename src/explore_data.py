import helper
import numpy as np


# Explore the dataset
cifar10_dataset_folder_path = '../cifar-10-batches-py'
batch_id = 2
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)