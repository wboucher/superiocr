""" This class just contains a PIL image for the document we want to read"""
class RawImage:
	def __init__(self, image):
		self.image = image

""" This class contains the cleaned image in numpy array form"""
class CleanedImage:
	def __init__(self, clean_image, raw_image):
		self.image = clean_image(raw_image)

"""
This class contains the segmented document, represented according to the following scheme
data is a list of lines
lines are a list of words
words are a list of characters
each character is a numpy array image of whatever size it was when we cropped/segmented it
"""
class SegmentedDocument:
	def __init__(self, segmenter, clean_image)
		self.data = segmenter.do_all(clean_image)

"""
This class contains the recognized document, represented according to the following scheme
data is a list of lines
lines are a list of words
words are a list of characters
each character is a single maximum likelihood estimate of its class
"""
class RecognizedDocument:
	def __init__(self, character_model, seg_doc)”
		self.data = [[[character_model.predict(c) for c in w] for w in l] for l in seg_doc.data]

"""
This class contains the final document, post processed, as a string
"""
class PostDocument:
	def __init__(self, post_processor, rec_doc):
		self.text = post_processor(rec_doc)
"""
This class holds functions which segments the image according to different algorithms,
based on a numpy array of the image, returning numpyified well-segmented images
"""
class Segmenter:
    def segment_lines():
        pass
    def segment_words():
        pass
    def segment_characters():
        pass
    def trim_characters():
        pass
    def do_all():
        pass

"""
The model class is the basic class for learning a model for mapping numpy arrays
to predicted characters.
"""
class Model:
    def __init__(self):
        pass
    # takes in a numpy array containining an unrolled image per row as X
    # and a list of single-character string labels for the y
    def train(self, X, y):
        pass
    # takes in an Nx(R*C) numpy array containing an image per row as X,
    # and returns a length N list containing the classifications for each image
    def predict(self, X):
        return [1 for row in X]
    # takes in an Nx(R*C) numpy array containing an image per row as X
    # and returns an Nx(52 + 10) numpy array containing probabilities for the
    # character being in that class
    def probs(self, X):
        pass
    # saves the model so that pickle can read it
    def save(self, filename):
        pickle.dump(self, file(filename, 'w'))
    # prints out the number it gets correct on a test set
    def correct(self, testX, testy):
        right = 0
        wrong = 0

        for X, y in zip(testX, testy):
            if self.predict(X) == y:
                right += 1
            else:
                wrong += 1

        print "Got ", right * 100.0 / (right + wrong), "% correct"
        return right * 1.0 * 100.0 / (right + wrong)


"""
#These classes could be used like this:

from Util import *
import PreProcessing
import CharacterSegmentation
import CharacterRecognition
import PostProcessing

raw = RawImage(read_image(filename))
clean = CleanedImage(current_cleaner, raw)
segmented = SegmentedDocument(current_segmenter, clean)
recognized = RecognizedDocument(current_model, segmented)
print PostDocument(current_processor, recognized)
"""
