
import numpy as np
from aucCalc import *
import progressbar
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from itertools import chain, combinations

import prune_duplicate
import names

def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


resultFile = '/home/mnz/results-2021-06-03.19_36_09.714042.txt-2'
resultFilePath = f'{resultFile}'
labels = pd.read_csv('./datasets/places365/places365_val.txt', delimiter=' ', header=None)
labels.columns = ['image', 'class']

conceptMat = np.load(resultFilePath+"-concept.npy")
if len(conceptMat.shape) == 1:
	conceptMat = conceptMat[:, np.newaxis]
targetMat = np.load(resultFilePath+"-target.npy")

labels = labels.iloc[:targetMat.shape[0]]


#necMat = np.zeros((conceptMat.shape[1], targetMat.shape[1]))

#barBase = progressbar.ProgressBar(max_value=targetMat.shape[1])
#for ti in range(0,targetMat.shape[1]):

	#t = targetMat[:, ti]
	#bar = progressbar.ProgressBar(max_value=conceptMat.shape[1])
	#necMat[:, ti] = calcAUC(1-conceptMat, t, lineStyle='-', plot=False)
	#barBase.update(ti)

#barBase.finish()
	
#np.save(resultFilePath+"-2-nec.npy", necMat)

necMat = np.load(resultFilePath+"-2-nec.npy")


BASIS_NUM = 7
nclasses = 365

# print the csv output

#for prediction_ind in range(nclasses):

	## name of the class
	#print(names.places_names[prediction_ind][1])

	#for a in [ t1 for t1 in necMat[:,prediction_ind].argsort()][::-1][:BASIS_NUM]:
		#print(necMat[a,prediction_ind], end=',')
	#print()
	
	#for a in [ t1 for t1 in necMat[:,prediction_ind].argsort()][::-1][:BASIS_NUM]:
		#print(names.concept_names[a], f" ({a})", end=',')
	#print()



# soccer field
prediction_ind = 310
conceptNos = [20,465,267,395,7,285,529]
#conceptNos = [465, 285, 529]


## filtering
##index = conceptMat[:,529] > .5
##index = ( (labels['class'] == prediction_ind) | (targetMat.argmax(axis=1) == prediction_ind) ) & \
#index = (labels['class'] != targetMat.argmax(axis=1))
#filenames = labels[index].reset_index()
#conceptMat = conceptMat[index,:]
#targetMat = targetMat[index,:]

#for i, row in filenames.iterrows():
	#plt.imshow(plt.imread('./datasets/places365/val_256/'+row['image']))
	#print([(conceptMat[i,conceptNo], names.concept_names[conceptNo]) for conceptNo in conceptMat[i,:].argsort()[::-1]])
	#print(targetMat[i,:].max(), names.places_names[targetMat[i,:].argmax()], names.places_names[row['class']], targetMat[i,row['class']])
	#plt.show()

t0 = 0.75
c0 = 0.5
for prediction_ind in range(660):
	
	#t = targetMat[:,prediction_ind]
	t = targetMat.argmax(axis=1)==prediction_ind
	
	clf = DecisionTreeClassifier(min_samples_leaf=2, max_depth=16)
	X_train = conceptMat[:, :]>c0
	y_train = t>t0
	clf.fit(X_train, y_train)
	#clf.classes_ = names.places_names
	
	filteredNames = [names.concept_names[i] for i in conceptNos]
	prune_duplicate.prune_duplicate_leaves(clf)
	text_representation = export_text(clf, feature_names=names.concept_names, max_depth=16)
	print(names.places_names[prediction_ind])
	print(text_representation)

##conceptMat = (conceptMat - np.min(conceptMat, axis=0))/(np.max(conceptMat, axis=0) - np.min(conceptMat, axis=0))
##targetMat = (targetMat - np.min(targetMat, axis=0))/(np.max(targetMat, axis=0) - np.min(targetMat, axis=0))

#t = targetMat[:,prediction_ind]

#[names.concept_names[conceptNo] for conceptNo in necMat[:, prediction_ind].argsort()]

#plt.imshow(conceptMat[t>t0,:][:, conceptNos])
#plt.show()

#plt.imshow(conceptMat[t<=t0,:][:, conceptNos])
#plt.show()

#codedP = np.matmul( conceptMat[t>t0,:][:, conceptNos]>c0 , np.array([2**np.arange(len(conceptNos))]).T)
#nP, _, _ = plt.hist(codedP, bins=range(129))
#plt.show()
#codedN = np.matmul( conceptMat[t<=t0,:][:, conceptNos]>c0 , np.array([2**np.arange(len(conceptNos))]).T)
#nN, _, _ = plt.hist(codedN, bins=range(129))
#plt.show()

#for i in range(128):
	#print([(i // 2**x)%2 for x in range(len(conceptNos))], nP[i], nN[i])


#plt.imshow(conceptMat[t.argsort(),:])
#plt.show()



##for conceptNo in np.min(powerset(conceptNos)):
#for conceptNo in conceptNos:
	#c = conceptMat[:,conceptNo]
	##c = np.min(conceptMat[:,conceptNo], axis=1)
	#out = calc4scores(c, t, plot=False)
	#print(conceptNo, out)
	#plotConceptTarget(c, t, .5, conceptName=names.concept_names[conceptNo], taskName=names.places_names[prediction_ind][1])
	#plt.title(conceptNo)
	#plt.show()


#print()

##soccer_field
##0.7549165188240801,0.6670935857183066,0.6207480168845126,0.5759862006632798,0.561765915478931,0.5204935776305916,0.501721606302598,
##object.grass  (20),object.pitch  (465),object.grandstand  (267),object.court  (395),object.person  (7),object_part.post  (285),object.goal  (529),

#t.argsort()
#conceptMat[t.argsort().reshape(-1),conceptNos[:7]]