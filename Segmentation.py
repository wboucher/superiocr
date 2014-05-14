import PIL
import Image
import numpy as np
import pickle

#input the name of the preprocessed file, minus the extension
#filename = raw_input("Enter the original filename, minus the png extension: ")
#filename_str = filename + "_bw.png"
#im = Image.open(filename_str)

def get_character_nps(im, filename):
    im2 = im.convert('L')
    np_im = np.asarray(im2)
    line_indices = []
    line_start = -1
    line_end = -1

    #calculate the indices of the lines where characters are
    for i in xrange(len(np_im)):
        sum = 0
        for j in xrange(len(np_im[0])):
            if np_im[i][j] == 0:
                sum = 1
                break
        if sum != 0 and line_start == -1:
            line_start = i
        elif sum == 0 and (line_start != -1):
            line_end = i
        if line_start != -1 and (line_end != -1):
            line_indices.append((line_start,line_end))
            line_start = -1
            line_end = -1

    #calculate the index of the box where each individual character is
    char_indices = []
    char_start = -1
    char_end = -1
    for i in xrange(len(line_indices)):
        for j in xrange(len(np_im[0])):
            sum = 0
            for k in xrange(line_indices[i][0],line_indices[i][1]):
                if np_im[k][j] == 0:
                    sum = 1
                    break
            if sum != 0 and char_start == -1:
                char_start = j
            elif sum == 0 and char_start != -1:
                char_end = j
            if char_start != -1 and char_end != -1:
                char_indices.append((line_indices[i],(char_start,char_end)))
                char_start = -1
                char_end = -1

    """
    print char_indices
    dims = [(index[1][0] - index[0][0], index[1][1]-index[0][1]) for index in char_indices]
    print dims
    characters = [[np.ones((dim[0]* 1.2, dim[0] * 1.2)) * 255] for dim in dims]
    for i in range(len(characters)):
        dim = dims[i]
        x_offset = dim[0] * .1 + ((dim[1] - dim[0])/2) 
        characters[i][dim[0]*.1:dim[0]*.1 + dim[0], x_offset:x_offset + dim[1]] = np.matrix(np_im[char_indices[i][0][0]:char_indices[i][1][0], char_indices[i][0][1]:char_indices[i][1][1]])

    characters = [np.asarray(Image.fromarray(c).resize((28.28))) for c in characters]
    """
    """
    #creates 28-by-28 boxes for each character, since that's what the
    #machine learning is trained on
    characters = [[[255 for i in range(28)] for j in range(28)]for char in char_indices]
    for i in range(len(char_indices)):
        for j in range(char_indices[i][0][0],char_indices[i][0][1]):
            for k in range(char_indices[i][1][0],char_indices[i][1][1]):
                characters[i][j-char_indices[i][0][0]+5][k-char_indices[i][1][0]+8] = np_im[j][k]
    """

    
    #creates 28-by-28 boxes for each character, since that's what the
    #machine learning is trained on
    characters = [char for char in char_indices]
    for i in range(len(char_indices)):
        for j in range(char_indices[i][0][0],char_indices[i][0][1]):
            for k in range(char_indices[i][1][0],char_indices[i][1][1]):
                characters[i] = np_im[char_indices[i][0][0]:char_indices[i][0][1], char_indices[i][1][0]:char_indices[i][1][1]]

    resized_characters = range(len(characters))
    for i in range(len(characters)):
        c        = characters[i]
        dim      = c.shape

        if dim[0] > dim[1]:
            y_margin = dim[0] * .1
            x_margin = (y_margin + ((dim[0] - dim[1]) / 2))
        else:
            x_margin = dim[1] * 0
            y_margin = (x_margin + ((dim[1] - dim[0]) / 2))

        

        this_char = 255 * np.ones((dim[0] + y_margin*2, dim[1] + x_margin *2))
        
        this_char[y_margin:y_margin + dim[0], x_margin:x_margin + dim[1]] = c
        resized_characters[i] = np.array(Image.fromarray(this_char).resize((28, 28)))
    
    
    #writes these 28-by-28 boxes to a file
    '''   outfile = filename + ".txt"
    with open(outfile,'w') as f:
        pickle.dump(characters,f)
    f.closed
 These boxes can be reclaimed as follows:
    with open(outfile,'r') as f:
	x = pickle.load(f)
	print x
    f.closed'''
    return (np.array([c.reshape(28 * 28) for c in resized_characters]), char_indices) #(np.array([np.array(c).reshape(28*28, 1) for c in characters])[:, :, 0], char_indices)
