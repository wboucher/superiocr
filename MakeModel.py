from Process_Data import *
from ReadData import *
from NeuralNetwork import *

images = join_data()
labels, _ = load_data()

all_caps_images = images[10160:36576]
all_caps_labels = images[10160:36576]

all_caps_images = 255 - all_caps_images

model = NNModel([TanhLayer(28*28 + 1, 250), TanhLayer(251, 36)], 36, .0001, .04)
model.train(all_caps_images, all_caps_labels, 1000)

pickle.dump(model, open('filename', 'w'))
