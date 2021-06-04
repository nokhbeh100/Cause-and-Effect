import torch
import progressbar
from auxLearn.auxLearn import CUDA
from matplotlib import pyplot as plt
import numpy as np

# this function is to be used as last function of model and concept
def doNothing(x):
    return x
 
def crossEvaluateMultiOutput(conceptModel, targetModel, actLoader, modelLastFunc=torch.nn.functional.softmax, conceptLastFunc=doNothing):
    #y: which class are we using to calculate the gradient
    evalPoints = []
    if CUDA:
        conceptModel = conceptModel.cuda()
    bar = progressbar.ProgressBar(max_value=len(actLoader))
        
    for i, (acts, classno) in enumerate(actLoader, 0):
        bar.update(i)
        if CUDA:
            acts = acts.cuda()
        conceptModel.eval()
        outputsConcept = conceptModel(acts)
        targetModel.eval()
        outputsModel = targetModel(acts)
        evalPoints.append( (conceptLastFunc(outputsConcept.reshape((1,-1))).detach().cpu().numpy(), 
                            modelLastFunc(outputsModel.reshape((1,-1))).detach().cpu().numpy() ) )
    bar.finish()
    return evalPoints
