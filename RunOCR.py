from Preprocessing  import *
from Segmentation   import *
from Postprocessing import *
import pickle


model  = pickle.load(open("all_characters_other.model", "r"))
infile = raw_input("Running SuperiOCR! \n")

print "Pre-processing image..."
pre_processed = preprocess(infile)
print "Image pre-Processed"

print "Segmenting image..."
(character_nps, char_indices) = get_character_nps(pre_processed, infile)
print "Image segmented"

def invert_all(image_nps):
    return 255 - image_nps

character_nps = invert_all(character_nps)

print "Classifying characters..."
classifications = model.predict(character_nps)

def to_character(classlabel):
    if classlabel < 11:
        return str(classlabel - 1)
    elif classlabel < 37:
        return chr(classlabel - 11 + 65)
    else:
        return chr(classlabel - 37 + 97)
        
classifications = [to_character(c+1) for c in classifications]
print "Characters classified"
out = postprocessing(classifications,char_indices)
print "Postprocessing..."
print "Writing to file..."
outfile_name = infile + ".txt"
f = open(outfile_name, 'w')
if len(out) > 0:
	f.write(out[0])
f.close()
f = open(outfile_name, 'a')
for i in range(len(out)-1):
	f.write(out[i+1])
f.close()
print "Done!"
