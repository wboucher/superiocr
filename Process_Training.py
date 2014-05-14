import scipy.io
import Segmentation
import Image
import numpy as np
import pickle
import os.path

def load_data():
    data      = scipy.io.loadmat("data/Characters/Lists/English/Fnt/lists.20.mat")
    labels    = data['list']['ALLlabels'][0,0]
    filenames = data['list']['ALLnames'][0,0]
    return labels, filenames

def get_segment(fname):
    return Segmentation.get_character_nps(Image.open(filename_prefix + fname + ".png").convert('L'), "")[0][0]

import multiprocessing as multi

def segment_data(filenames, labels, fname = 'data/image_np_chunk', start = 0, end = 21):
    filename_prefix = "data/Characters/English_Font/Fnt/"
    size = 3099
    for i in range(21)[start:end]:
        print i * 5, "%"
        p = multi.Pool(processes=3)
        chunk = p.map(get_segment, filenames[i*size:(i+1)*size], 1033)
        pickle.dump(chunk, open(fname + str(i) + '.data', 'w'))
        p.close()

    print "Done!"

def join_data(filename='data/image_np_chunk', range_size=21):
    ans = []
    for i in range(range_size):
        if os.path.exists(filename + str(i) + '.data'):
            ans.extend(pickle.load(open(filename + str(i) + '.data', 'r')))
    return np.matrix(ans)
    


