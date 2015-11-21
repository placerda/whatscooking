import numpy as np

'''
This module contains the functions needed to classify a list of receipes,
based on naive bayes method trained by a dataset of categorized receipes.
Some functions are based on the samples of the Machine Learning in Action book.
'''

def createVocabulary(trainReceipes):
	vocabSet = set([])
	for document in trainReceipes:
		vocabSet = vocabSet | set(document['ingredients'])
	return list(vocabSet)

def createClasses(trainReceipes):
	classSet = set([])
	for receipe in trainReceipes:
		classSet = classSet | set([receipe['cuisine']])
	return list(classSet)

def createFeatVector(vocabList, ingredientsList):
	featureVec = [0]*len(vocabList)
	notFound = 0
	notFoundTokens = []
	for word in ingredientsList:
		if word in vocabList:
			featureVec[vocabList.index(word)] = 1;
		else: 
			#print "word %s is not in my vocabulary" % word
			notFoundTokens.append(word)
			notFound += 1
	if notFound > 0:
		print " Receipe with ingrendients not found in the vocabulary"
		print "  tokens: %d, notfound: %d" % (len(ingredientsList),notFound)
		print "  not found: " + str(notFoundTokens)

	return featureVec, notFound

def trainNB(trainReceipes, vocabulary, classes):
	''' Bayes: p(c|w) = (p(w|c) * p(c)) / p(w) 
		This funcion calculates p(w|c) and p(c)
	'''
	''' Initialization '''
	# initialize numerator with 1 because some probability may be 0
	numeratorPwc = np.array([[1.0]*len(vocabulary)]*len(classes))

	# initialize denominator of each token with the number of 
	# classes because of the numerator 1 initialization
	denominatorPwc = np.array([len(classes)]*len(vocabulary))
	pc = np.array([0.0] * len(classes))	

	'''Calculates p(c) and p(w|c) vector for each class'''
	for receipe in trainReceipes:
		#pc
		pc[classes.index(receipe['cuisine'])] += 1
		#pwc
		receipeFeatVector, nfTokens = createFeatVector(vocabulary,receipe['ingredients'])
		numeratorPwc[classes.index(receipe['cuisine'])] += receipeFeatVector
		denominatorPwc += receipeFeatVector
	
	# calculates each class probability
	pc = pc / float(sum(pc))
	
	# using log to avoid underflow problem when doing multiplication
	pwc = np.log(numeratorPwc / denominatorPwc)
	return pc, pwc

def classifyNB(pc, pwc, ingredFeatVector):
	'''Calculates p(c|w) vector for each class'''
	pcw=np.array([0.0]*len(pwc))
	for i in range(len(pcw)):
		pcw[i] = sum(pwc[i] * ingredFeatVector) + np.log(pc[i])
	#get max p(c|w) value 
	classIndex = pcw.tolist().index(max(pcw))
	return classIndex
	
def run(trainReceipes, unkReceipes):
  	print "### run summary ###" 
	print ">train dataset size: %d"  %len(trainReceipes)
	print ">receipes to classify: %d"  %len(unkReceipes)	

	vocabulary = createVocabulary(trainReceipes)
	print "creating vocabulary..."
	print ">vocabulary size: %d"  %len(vocabulary)

  	classes = createClasses(trainReceipes)
	print "extracting classes..."
	print ">number of classes: %d"  %len(classes)

	print "training NB Classifier..."
  	pc, pwc = trainNB(trainReceipes, vocabulary, classes)
  	
  	print "classifying using NB..."
  	classReceipes = {}
  	nfTokens = 0
  	nfTokensReceipes = 0
  	for receipe in unkReceipes:
  		featureVector, nft = createFeatVector(vocabulary,receipe['ingredients'])
  		nfTokens += nft
  		if nft > 0: nfTokensReceipes += 1
  		classIndex= classifyNB(pc,pwc,featureVector)
  		classReceipes[receipe['id']] = classes[classIndex]
  	print ">ingrendients not in the vocabulary: %d" % nfTokens
  	print ">receipes with ingredients not in the vocabulary: %d" % nfTokensReceipes
  	return classReceipes