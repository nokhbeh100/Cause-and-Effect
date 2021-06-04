
import numpy as np
from aucCalc import *
import progressbar


resultFile = 'results-2021-06-03.19_36_09.714042.txt-2'
resultFilePath = f'{resultFile}'

conceptMat = np.load(resultFilePath+"-concept.npy")
if len(conceptMat.shape) == 1:
	conceptMat = conceptMat[:, np.newaxis]
targetMat = np.load(resultFilePath+"-target.npy")


necMat = np.zeros((conceptMat.shape[1], targetMat.shape[1]))

barBase = progressbar.ProgressBar(max_value=targetMat.shape[1])
for ti in range(0,targetMat.shape[1]):

	t = targetMat[:, ti]
	bar = progressbar.ProgressBar(max_value=conceptMat.shape[1])
	necMat[:, ti] = calcAUC(1-conceptMat, t, lineStyle='-', plot=False)
	barBase.update(ti)

barBase.finish()
	
np.save(resultFilePath+"-2-nec.npy", necMat)

#necMat = np.load(resultFilePath+"-2-nec.npy")

import names

BASIS_NUM = 7
nclasses = 365

for prediction_ind in range(nclasses):

	print(names.places_names[prediction_ind][1])

	for a in [ t1 for t1 in necMat[:,prediction_ind].argsort()][::-1][:BASIS_NUM]:
		print(necMat[a,prediction_ind], end=',')
	print()
	
	for a in [ t1 for t1 in necMat[:,prediction_ind].argsort()][::-1][:BASIS_NUM]:
		print(names.concept_names[a], end=',')
	print()
