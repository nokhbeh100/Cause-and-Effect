import datetime

import os.path
homedir = os.path.expanduser("~")
results = f'{homedir}/results-{datetime.datetime.now()}.txt'
results = results.replace(':','_').replace(' ','.')
print(results)


savemodels = ''

motherName = './models/resnet18_places365.pth.tar'
conceptName = './models/14.pth'
trainFolder = ''
conceptFolder = ''
evaluationFolder = './datasets/places365/'

#motherName = './mothernet-colorized_0.pt'
#trainFolder = './datasets/MNIST_colorize_complex_0.0/'
#conceptFolder = './datasets/MNIST_shuffled_colorize_complex_0.0/'
#evaluationFolder = './datasets/MNIST_colorize_complex_0.5/'

N_EPOCHS_TRAIN = N_EPOCHS = 50
smart_stop = 5
d = 3 # number of channels of input images
TRAIN_MODEL = False
TRAIN_CONCEPT = False

logFile = open(results, 'w')
logFile.write(f"forgettingNets:\n")

if TRAIN_MODEL:
    logFile.write(f"train folder: {trainFolder}\n")
else:
    logFile.write(f"model being tested: {motherName}\n")

if TRAIN_CONCEPT:
    logFile.write(f"concept: {conceptFolder}\n")
else:
    logFile.write(f"concept model being used: {conceptName}\n")

logFile.write(f"evaluation points: {evaluationFolder}\n")


import torch    
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import *
import numpy as np
from matplotlib import pyplot as plt
from broden import brodenDataset
import names

import torchvision.transforms as transforms

from auxLearn.auxLearnVision import *
from auxLearn.auxLearn import *
from analysisTools import *

print ('testing CUDA:', CUDA)



#%%

# both models must be converted to sequential models for better spliting points
referenceModelBase = torchvision.models.__dict__['resnet18'](num_classes=365)
L1 = getModules(referenceModelBase)
referenceModel = nn.Sequential( *L1 )


motherNetBase = torchvision.models.__dict__['resnet18'](num_classes=365)
L2 = getModules(motherNetBase)
motherNet = nn.Sequential( *L2 )

if CUDA:
    motherNet = motherNet.cuda()
    referenceModel = referenceModel.cuda()
    

criterion = nn.BCELoss()
mseloss = nn.MSELoss()


if TRAIN_MODEL:
    #in case of training is required:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,)*d, (0.5,)*d)])
    
    trainLoader, evalLoader, _ = loadDataset(trainFolder, transform=transform, num_workers=0)
    
    optimizer = optim.SGD(motherNet.parameters(), lr=0.001, momentum=0.9)
    trainForEpoches(motherNet, trainLoader, evalLoader, optimizer, criterion, N_EPOCHS = N_EPOCHS_TRAIN)
    
    # save model as the reference
    torch.save(motherNet.state_dict(), motherName)


# if model is pretrained
checkpoint = torch.load(motherName, map_location='cpu')
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
referenceModelBase.load_state_dict(state_dict)
motherNetBase.load_state_dict(state_dict)

#%%


ts = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])


#%%

print('Inspecting layers')

#labels = open(labels_path).read().splitlines()
#trying to build up from last to first

conceptValidLayersAcc = []
conceptValidLayersSen = []
layers = list( motherNet.children() )

for concept_no, concept_name, cats in names.concept_tuples:
    for layerNo in [2]:
        
        print (f'inspecting: {-layerNo} against concept: {concept_no}{concept_name}')
        print(f'getting ready training for the layerNo = {layerNo}')
    
            
        if len(list(layers[-layerNo+1].parameters())) == 0:
            print('skipping due to nontrainable next layer (maybe activation/dropout/flatten)')
            continue
        
        # reset the network
        motherNetBase.load_state_dict(state_dict)
                
        # after the layer under inspection
        conceptModel = singleOutput( nn.Sequential(*list(motherNet.children())[-layerNo+1:]) )
        
        # after the layer under inspection
        targetModel = nn.Sequential(*list(referenceModel.children())[-layerNo+1:])
        
        # before the layer under inspection
        firstSection = nn.Sequential(*list(referenceModel.children())[:-layerNo+1])
                

        transform = transforms.Compose(
            [gray2rgb,
             transforms.ToTensor(),
             #transforms.Normalize((0.5,)*d, (0.5,)*d),
             ts,
             sampleToBatch,
             toCuda,
             firstSection])
        
        if TRAIN_CONCEPT:
            #conceptTrainLoader, conceptTestLoader, conceptValidLoader = loadDataset(conceptFolder, transform=transform, num_workers=0)
            broden = brodenDataset('./datasets/broden1_224/', transform, resize=(7, 7))    
            
        #%%
        # this is required to get activation of the reference model by toActs function
        evalSet = torchvision.datasets.ImageFolder(evaluationFolder, loader=plt.imread, transform=transform )
        evalLoader = torch.utils.data.DataLoader(evalSet, batch_size=1, shuffle=False, num_workers=0)

        
        if TRAIN_CONCEPT:
            conceptTrainLoader = data.DataLoader(broden.get_train_concept(concept_no), num_workers=0)
            conceptTestLoader = data.DataLoader(broden.get_valid_concept(concept_no), num_workers=0)
            
            #optimizer = optim.SGD(conceptModel.parameters(), lr=0.001, momentum=0.9)
            optimizer = optim.Adam(conceptModel.parameters())
                    
            #train to retrieve information
            print(f'training for the layerNo = {layerNo}')
            logFile.write(f"conceptModel:{conceptModel.__class__.__name__}, layerNo:{layerNo}, optimizer: {optimizer.__class__.__name__}, N_EPOCHS:{N_EPOCHS}, smart_stop:{smart_stop}\n")
            trainForEpoches(conceptModel, conceptTrainLoader, conceptTestLoader, optimizer, criterion, N_EPOCHS = N_EPOCHS, smart_stop=smart_stop, resultFile=logFile, hardNeg=True)
            
            if savemodels:
                torch.save(conceptModel.state_dict(), savemodels+f'concept-{conceptModel.__class__.__name__}-layer{layerNo}.pt')
            
            
            print(f'ready for validation layerNo = {layerNo}')
            
            valid_loss, valid_acc = evaluate(conceptModel, conceptTestLoader, criterion)
            conceptValidLayersAcc.append(valid_acc)
            
            print('BCE loss:')
            print(f"valid_acc={valid_acc}")
            print(f"valid_loss={valid_loss}")
            logFile.write(f"BCE loss\n")
            logFile.write(f"valid_acc={valid_acc}\n")
            logFile.write(f"valid_loss={valid_loss}\n")
            
            valid_loss, valid_acc = evaluate(conceptModel, conceptTestLoader, mseloss)
            
            print('MSE loss:')
            print(f"valid_acc={valid_acc}")
            print(f"valid_loss={valid_loss}")
            logFile.write(f"MSE loss\n")
            logFile.write(f"valid_acc={valid_acc}\n")
            logFile.write(f"valid_loss={valid_loss}\n")
            
            
            logFile.flush()
        else:
            conceptModel = SingleSigmoidFeatureClassifier()
            conceptModel.load_state_dict(torch.load(conceptName))
            #motherNet.load_state_dict(state_dict)
        
        
        print(f'evaluating sensetivity for layerNo = {layerNo}')
        evalPoints = crossEvaluateMultiOutput( conceptModel, targetModel, evalLoader)
        cs, ts = zip(*evalPoints)
        conceptMat = np.concatenate(cs, axis=0)
        targetMat = np.concatenate(ts, axis=0)
        conceptFilename = f"{results}-{concept_no}-{layerNo}-concept.npy"
        targetFilename = f"{results}-{concept_no}-{layerNo}-target.npy"
        
        logFile.write(f"writng concept to {conceptFilename}\n")
        np.save(conceptFilename, conceptMat)
        logFile.write(f"writng target to {targetFilename}\n")
        np.save(targetFilename, targetMat)
        logFile.flush()
        
    
    

print("conceptValidLayersAcc=", conceptValidLayersAcc)    
#plt.plot(conceptValidLayersAcc)
    
logFile.close()
