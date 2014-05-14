from PIL import Image
from pylab import *
import numpy as np

# This function takes in a numpy array encoding an image as a greyscale single
# row vector, reshapes it, and gives you a PIL image, which can be shown using
# .show()
def image_from_np_row(row, shape=(28,28)):
    arr = row.reshape(shape).astype(np.uint8)
    return Image.fromarray(arr)

def image_to_np(image):
    return np.asarray(image).convert("l")
