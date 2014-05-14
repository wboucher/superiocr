from ReadData import *
from Display import *
from ConvolutionalNetwork import *

model = NNModel([TanhLayer(28 * 28 + 1, 49), TanhLayer(50, 10)], .0001, .05)
model.train(images[:100], labels[:100])
