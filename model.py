import torch
from torchvision.models import *
import torch.nn.functional as F
from torch import nn
import numpy as np
import gc
from auxLearn.auxLearnVision import CUDA


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# this class is borrowed from IBD article implementation for comparison
class SingleSigmoidFeatureClassifier(torch.nn.Module):
    def __init__(self, feature=None, layer=None, fo=None):
        super(SingleSigmoidFeatureClassifier, self).__init__()
        #self.fc = IndexLinear(1024, 660)
        self.fc = torch.nn.Linear(512, 660)
        self.sig = torch.nn.Sigmoid()


    def forward(self, input):
        return self.sig(self.fc(input.squeeze()))        
        
# just convert the model to single output!            
class singleOutput(torch.nn.Module):
    def __init__(self, model):
        super(singleOutput, self).__init__()
        self.model = model


    def forward(self, inputs):
        return torch.sigmoid(self.model(inputs)[:,0].to(torch.float64))        
            
    
# following helper functions help us to convert complex models to sequential models for easier breakdown at any layer

# helper function for resnets
def getResnetModules(model):
    L1 = [model.conv1, model.bn1, model.relu, model.maxpool] + list(model.layer1.children()) + list(model.layer2.children()) + list(model.layer3.children()) + list(model.layer4.children()) + [model.avgpool, nn.Flatten(1), model.fc]
    return L1

# helper function for vggs
def getVggModules(model):
    L1 = list(model.features.children()) + [model.avgpool, nn.Flatten(1)] + list(model.classifier.children())
    return L1
    
# helper function for inception_v3
def getInceptionV3Modules(model):
    L1 = [model.Conv2d_1a_3x3,
    model.Conv2d_2a_3x3, model.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2), 
    model.Conv2d_3b_1x1,
    model.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
    model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
    model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d, model.Mixed_6e,
    model.Mixed_7a, model.Mixed_7b, model.Mixed_7c, nn.AdaptiveAvgPool2d((1,1)), nn.Dropout(inplace=True), nn.Flatten(1),
    model.fc]
    return L1    
    
# helper function for alexNet    
def getAlexnetModules(model):
    L1 = list( model.features.children() ) + list( model.avgpool.children() ) + [nn.Flatten(1)] + list( model.classifier.children() )
    return L1

#  TODO: an abtraction of all above functions
def getModules(model):
    if type(model) is ResNet:
        return getResnetModules(model)
    if type(model) is VGG:
        return getVggModules(model)
    if type(model) is Inception3:
        return getInceptionV3Modules(model)
    if type(model) is AlexNet:
        return getAlexnetModules(model)
    
    return None