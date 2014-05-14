import numpy as np

def no_gray(image, threshold = 128):
    return 256 * np.array(image > threshold, dtype=np.uint8)
