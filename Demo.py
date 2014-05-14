# this OCR project requires
# the following Python 2.7 packages:
# numpy, scipy, matplotlib, PIL, sklearn

print "Training model..."

from Display import *
from ReadData import *
from CharacterRecognition import *

model = LogisticModel()
model.train(images[:1000], labels[:1000])
model.correct(images[1000:2000], labels[1000:2000])

f = open('testoutput.txt', 'w')
f.truncate()


# basic postprocessing is included here
# easier to do postprocessing here in the recognition function itself
def compare(i):
    model_prediction = model.predict(images[i])
    print "The model thinks it should be: ", model_prediction
    model_prediction.tofile(f, sep=",", format="%s")
    print "The image is..."
    image_from_np_row(images[i]).show()

compare(13)
f.close()
