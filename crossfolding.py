#!/usr/bin/env python
import bayes, sys, json, numpy as np, logging, argparse
'''
Program to evaluate classifier based on ten fold validation
'''

#setup argparse
parser = argparse.ArgumentParser(
    description='Whatscooking Program cross folding classifier'
)
parser.add_argument("trainRecipesFile", help="training data set")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.INFO)

def crossfolding(trainReceipes):
	logging.info("### run summary ###")
	#initialize counters (at least 2)
	numPartitions = 10
	accuracy = np.array([0.0] * numPartitions)
	partitionsSize = np.array([0] * numPartitions)
	
	#define partitions size
	if len(trainReceipes) < numPartitions: 
		logging.error("Train dataset must have more than %d items" % numPartitions)
		sys.exit(0)
	partitionsSize += len(trainReceipes) / numPartitions
	for i in range(len(trainReceipes) % numPartitions):
		partitionsSize[i] += 1
	logging.info(">number of training receipes: %d" % len(trainReceipes))

	#calculate accuracy for each partition
	logging.info("...calculating accuracy for each partition...")
	partitionIndex = 0
	for i in range(numPartitions):
		logging.info("FOLD %d" % (i+1))
		#get train and test lists		
		testList = trainReceipes[partitionIndex:partitionIndex+partitionsSize[i]]
		trainList = [] * (len(trainReceipes)-len(testList))
		for nDocument in range(len(trainReceipes)):
			if (nDocument < partitionIndex) | (nDocument>partitionIndex+partitionsSize[i]):
				trainList.append(trainReceipes[nDocument])
		partitionIndex += partitionsSize[i]
		
		#classify test list
		classifiedList = bayes.run(trainList,testList)
		totalReceipes = 0.0
		truePositives = 0.0
		for receipe in testList:
			totalReceipes += 1
			if classifiedList[receipe['id']] == receipe['cuisine']:
				truePositives += 1
		#compare classification to calculate accuracy
		accuracy[i] = truePositives / totalReceipes
	#calculate avg accuracy
	avgAccuracy = 0.0
	avgAccuracy = np.average(accuracy)
	return avgAccuracy

def main(argv):
	with open(args.trainRecipesFile) as train_receipes_file:
		trainJson = json.load(train_receipes_file)

	#does crossfolding validation
	accuracy = crossfolding (trainJson)
	print "Accuracy=%f" % accuracy

if __name__ == '__main__':
    main(sys.argv)