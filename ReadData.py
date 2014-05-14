# Adapted from cvxopt.org/_downloads/mnist.py by Aaron Tucker

import os, struct
from array import array
import numpy as np
import Image
import scipy.io
import Segmentation

def read_mnist(dataset="train"):
    image_fname = os.getcwd() + "/data/MNIST/" + dataset + "-images.idx3-ubyte"
    label_fname = os.getcwd() + "/data/MNIST/" + dataset + "-labels.idx1-ubyte"

    label_file = open(label_fname, 'rb')
    magic_nr , size = struct.unpack(">II", label_file.read(8))
    label_bytes = array("b", label_file.read())
    label_file.close()

    image_file = open(image_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", image_file.read(16))
    image_bytes = array("B", image_file.read())
    image_file.close()

    ind = [ k for k in xrange(size)]
    images = []
    labels = []
    for i in xrange(len(ind)):
        images.append(image_bytes[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels.append(label_bytes[ind[i]])

    return np.matrix(images), labels

def read_image(filename):
    return Image.open(filename)

def invert_all(image_nps):
    return 255 - image_nps

def read_character_examples():
    data      = scipy.io.loadmat("data/Characters/Lists/English/Fnt/lists.20.mat")
    labels    = data['list']['ALLlabels'][0,0]
    filenames = data['list']['ALLnames'][0,0]
    filename_prefix = "data/Characters/English_Font/Fnt/"

    images = [Segmentation.get_character_nps(Image.open(filename_prefix + fname + ".png").convert('L'), "")[0][0] for fname in filenames]

    return invert_all(np.matrix(images)), np.matrix(labels)

images, labels = read_character_examples()
