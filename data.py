import os
import sys
from urllib.request import urlretrieve
import gzip
import struct
import array
import numpy as np
import _pickle as pickle
import pyalexutil as alxu


# ========================================================================
# ====== Next two functions copied from autograd/examples/data_mnist.py ==
# ====== Maclaurin, Duvenaud, Johnson (MIT License) ======================

def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


# ========================================================================

def load_mnist(folder, filename='mnist.pickle'):
    folder = os.path.expanduser(folder)
    assert os.path.exists(folder), "folder {} does not exist!".format(folder)
    fullfile = alxu.sys.fullfile(folder, filename)

    if os.path.isfile(fullfile):
        with open(fullfile, 'rb') as f:
            return pickle.load(f)
    else:
        yesno = input("Data not found in specified directory. Download (y/n?)")
        if yesno.upper()[0] == 'Y':
            print('{} Downloading files...'.format(alxu.sys.current_timestamp()))
            train_images, train_labels, test_images, test_labels = mnist()
            with open(fullfile, 'wb') as f:
                print('{} Saving results...'.format(alxu.sys.current_timestamp()), end='')
                pickle.dump([train_images, train_labels, test_images, test_labels], f)
                print('Done!')
                return train_images, train_labels, test_images, test_labels
