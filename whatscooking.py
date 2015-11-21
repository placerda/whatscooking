#!/usr/bin/env python

import bayes, sys, json

def main(argv):
	if len(argv) > 2:
		#load json files.
		with open(argv[1]) as train_receipes_file:
			trainJson = json.load(train_receipes_file)
		with open(argv[2]) as unknown_receipes_file:
			unknownJson = json.load(unknown_receipes_file)

		results = {}

		#run naive bayes classifier.
		results = bayes.run(trainJson,unknownJson)

		# write to output file
		text_file = open("../output/submission.csv", "w")
		text_file.write('id,cuisine\n')
		for i in results:
			text_file.write(str(i) + ',' + results[i] + '\n')
		text_file.close()

	else:
		print("use: whatscooking.py {trainRecipesFile} {unknownRecipesFile}")

if __name__ == '__main__':
    main(sys.argv)