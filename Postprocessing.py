# Output recognized characters to text file
# Reach goal: compare characters to a lexicon with best fitting words


def postprocessing(matches, indices):
	current = indices[0]
	output = []
	for i in range(len(matches)):
		if current[0][0] != indices[i][0][0] or indices[i][1][0] - current[1][1] >= 10:
			output.append(" ")
		current = indices[i]
		output.append(matches[i])
	return output


