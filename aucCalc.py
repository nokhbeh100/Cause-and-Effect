import numpy as np
import matplotlib.pyplot as plt


def calculateROCF1(c, t):
	''' x is the thresholds
	y is the F1 score of that threshold '''
	x = np.arange(0,1.01,.01)#[0]+np.sort(th).tolist()+[1]
	if len(t.shape) == 1:
		t = t.reshape(-1,1)
	if len(c.shape) == 1:
		c = c.reshape(-1,1)
	y = []
	for th in x:
		TP = np.sum( (c> th) & (t<=th), axis=0)
		TN = np.sum( (c<=th) & (t> th), axis=0)
		F  = np.sum( (c> th) & (t> th), axis=0)
		Sen  = TP / (TP + F)
		Sen[(TP == 0)  & (F == 0)] = 1.
		Spec = TN / (TN + F)
		Spec[(TN == 0)  & (F == 0)] = 1.
		f1 = 2 * Sen * Spec / (Sen + Spec)
		f1[np.isnan(f1)] = 0
		y.append( f1 )        
		
	return x, y



def calculateROCAcc(c, t):
	''' x is the thresholds
	y is the accuracy of that threshold '''
	th = np.minimum(c, t)
	x = [0]+np.sort(t).tolist()+[1]
	y = [0]+(np.array(range(th.shape[0]))/th.shape[0]).tolist()+[1]
	return x, y


def calcAUC(c, t, lineStyle='-', plot=True):
	# designed for neg nec
	x, y = calculateROCF1(c, t)

	if plot:
		plt.plot(x, y, lineStyle)
		plt.xlabel('Threshold')
		plt.ylabel('F1 score')
	#plt.show()


	x0 = 0
	y0 = 0
	AUC = 0
	for x1, y1 in zip(x, y):
		thisArea = (x1-x0)*(y1+y0)/2
		AUC += thisArea
		x0, y0 = x1, y1
		#print(x1, y1, thisArea, AUC)
	return AUC

def plotConceptTarget(c, t, th, ms=4, conceptName='Concept', taskName='Task'):
	plt.plot(c, t, 'x', markersize=ms)
	plt.plot([th, th], [0, 1], 'r')
	plt.plot([0, 1], [1-th, 1-th], 'r')
	c1 = sum((c > 1-th) & (t > 1-th))
	plt.text(.5+th/2, 1-th/2, f'{c1}')
	c2 = sum((c <= th) & (t > 1-th))
	plt.text(th/2, 1-th/2, f'{c2}')
	c3 = sum((c > 1-th) & (t <= th))
	plt.text(.5+th/2, (1-th)/2, f'{c3}')
	c4 = sum((c <= th) & (t <= th))
	plt.text(th/2, (1-th)/2, f'{c4}')
	
	plt.xlabel(conceptName)
	plt.ylabel(taskName)
	
	print([c1,c2,c3,c4])
	
	
def calc4scores(c, t, plot=True):
	
	nec = calcAUC(1-c, t, lineStyle='-', plot=plot)
	#print(f"X necessary AUC:{nec}")
	
	suff = calcAUC(c, 1-t, lineStyle='--', plot=plot)
	#print(f"X sufficient AUC:{suff}")
	
	negnec = calcAUC(c, t, lineStyle='-.', plot=plot)
	#print(f"neg X necessary AUC:{negnec}")
	
	negsuff = calcAUC(1-c, 1-t, lineStyle=':', plot=plot)
	#print(f"neg X sufficient AUC:{negsuff}")
	
	if plot:
		plt.legend(['necessary', 'sufficient', 'neg necessary', 'neg sufficient'])
		
	return nec, suff, negnec, negsuff