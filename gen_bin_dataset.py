import os
import numpy as np
import h5py
import cv2


TRAIN_dir = "./TRAIN_data_224_v8"
TEST_dir = "./TEST_data_224_v8"
HDF5_PATH = "./dataset-ariel10-224x224x3-FRS-GBCC.hdf5"
TGT_SZ = (224, 224)


def map_addrs_labels(dir):
    labels = os.listdir(dir)
    labels.sort()
    label_to_idx = {labels[i]: i for i in range(len(labels))}
    return label_to_idx


def count_samples(dir):
    cnt = 0
    for d in os.listdir(dir):
        cnt += len(os.listdir(os.path.join(dir, d)))
    return cnt


def get_dataset_shape(num_samples):
    return num_samples, TGT_SZ[0], TGT_SZ[1], 3


def init_hdf5_dataset(train_shape, test_shape):
    hdf5_file = h5py.File(HDF5_PATH, mode='w')
    hdf5_file.create_dataset("X_train", train_shape, np.uint8, compression="gzip", compression_opts=9, chunks=True)
    hdf5_file.create_dataset("y_train", (train_shape[0],), np.int8, compression="gzip", compression_opts=9, chunks=True)
    hdf5_file.create_dataset("X_test", test_shape, np.uint8, compression="gzip", compression_opts=9, chunks=True)
    hdf5_file.create_dataset("y_test", (test_shape[0],), np.int8, compression="gzip", compression_opts=9, chunks=True)
    return hdf5_file


def fill_hdf5_dataset(hdf5_file, dir, X_dataset, y_dataset, label_to_idx):
    dirs = os.listdir(dir)
    i = 0
    for d in dirs:
        sub_dir = os.path.join(dir, d)
        files = os.listdir(sub_dir)
        for f in files:
            print(i)
            img_path = os.path.join(sub_dir, f)
            img = cv2.imread(img_path)
            img = cv2.resize(img, TGT_SZ, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            hdf5_file[X_dataset][i, ...] = img
            hdf5_file[y_dataset][i, ...] = label_to_idx[d]
            i += 1
    return hdf5_file


train_idx = map_addrs_labels(TRAIN_dir)
train_sz = count_samples(TRAIN_dir)
train_shape = get_dataset_shape(train_sz)

test_idx = map_addrs_labels(TEST_dir)
test_sz = count_samples(TEST_dir)
test_shape = get_dataset_shape(test_sz)

hdf5_file = init_hdf5_dataset(train_shape, test_shape)

hdf5_file = fill_hdf5_dataset(hdf5_file, TRAIN_dir, "X_train", "y_train", train_idx)
hdf5_file = fill_hdf5_dataset(hdf5_file, TEST_dir, "X_test", "y_test", test_idx)

hdf5_file.close()
print("Terminated")
