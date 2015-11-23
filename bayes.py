import numpy as np
import logging

'''
This module contains the functions needed to classify a list of recipes,
based on naive bayes method trained by a dataset of categorized recipes.
Some functions are based on the samples of the Machine Learning in Action book.
'''

def createVocabulary(trainrecipes):
	vocabSet = set([])
	for document in trainrecipes:
		vocabSet = vocabSet | set(document['ingredients'])
	return list(vocabSet)

def createClasses(trainrecipes):
	classSet = set([])
	for recipe in trainrecipes:
		classSet = classSet | set([recipe['cuisine']])
	return list(classSet)

def createFeatVector(vocabList, ingredientsList):
	featureVec = [0]*len(vocabList)
	notFound = 0
	notFoundTokens = []
	for word in ingredientsList:
		if word in vocabList:
			featureVec[vocabList.index(word)] = 1;
		else: 
			logging.debug("word %s is not in my vocabulary" % word)
			notFoundTokens.append(word)
			notFound += 1
	if notFound > 0:
		logging.debug(" recipe with ingrendients not found in the vocabulary")
		logging.debug("  tokens: %d, notfound: %d" % (len(ingredientsList),notFound))
		logging.debug("  not found: " + str(notFoundTokens))

	return featureVec, notFound

def trainNB(trainrecipes, vocabulary, classes):
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
	for recipe in trainrecipes:
		#pc
		pc[classes.index(recipe['cuisine'])] += 1
		#pwc
		recipeFeatVector, nfTokens = createFeatVector(vocabulary,recipe['ingredients'])
		numeratorPwc[classes.index(recipe['cuisine'])] += recipeFeatVector
		denominatorPwc += recipeFeatVector
	
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
	
def run(trainrecipes, unkrecipes):

	logging.info("creating vocabulary...")
	vocabulary = createVocabulary(trainrecipes)
	
  	logging.info("extracting classes...")
	classes = createClasses(trainrecipes)
	
	logging.info("training NB Classifier...")
  	pc, pwc = trainNB(trainrecipes, vocabulary, classes)
  	
  	logging.info("classifying using NB...")
  	classrecipes = {}
  	nfTokens = 0
  	nfTokensrecipes = 0
  	for recipe in unkrecipes:
  		featureVector, nft = createFeatVector(vocabulary,recipe['ingredients'])
  		nfTokens += nft
  		if nft > 0: nfTokensrecipes += 1
  		classIndex= classifyNB(pc,pwc,featureVector)
  		classrecipes[recipe['id']] = classes[classIndex]

  	logging.info("### NB run summary ###")
	logging.info("  train dataset size: %d"  %len(trainrecipes))
	logging.info("  recipes to classify: %d"  %len(unkrecipes))
	logging.info("  vocabulary size: %d"  %len(vocabulary))
	logging.info("  number of classes: %d"  %len(classes))
  	logging.info("  ingredients not in the vocabulary: %d" % nfTokens)
  	logging.info("  recipes with ingredients not in the vocabulary: %d" % nfTokensrecipes)
  	
  	return classrecipes