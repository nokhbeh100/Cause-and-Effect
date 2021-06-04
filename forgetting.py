
from settings import *


print(logFilePath)


logFile = open(logFilePath, 'w')
logFile.write(f"forgettingNets:\n")

if TRAIN_MODEL:
    logFile.write(f"train folder: {trainFolder}\n")
else:
    logFile.write(f"model being tested: {motherName}\n")

if TRAIN_CONCEPT:
    if USE_BRODEN:
        logFile.write(f"broden concept_no: {concept_no}\n")
    else:        
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

import torchvision.transforms as transforms

from auxLearn.auxLearnVision import *
from auxLearn.auxLearn import *
from analysisTools import *

print ('testing CUDA:', CUDA)



#%%

# both models must be converted to sequential models for better spliting points
referenceModelBase = torchvision.models.__dict__['resnet18'](num_classes=365)
referenceModel = nn.Sequential( *getModules(referenceModelBase) )


motherNetBase = torchvision.models.__dict__['resnet18'](num_classes=365)
motherNet = nn.Sequential( *getModules(motherNetBase) )

if CUDA:
    motherNet = motherNet.cuda()
    referenceModel = referenceModel.cuda()
    

criterion = nn.BCELoss()


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
motherNet = nn.Sequential( *getModules(motherNetBase) )
motherNetBase.load_state_dict(state_dict)
referenceModel = nn.Sequential( *getModules(referenceModelBase) )


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

for layerNo in [2]:
    
    print (f'inspecting: {-layerNo}')
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
    
    # before the layer under inspection, remember to drop avg pooling when training with BRODEN
    firstSection = nn.Sequential(*list(referenceModel.children())[:-layerNo+1])
    firstSection.eval()
                    

    transform = transforms.Compose(
        [gray2rgb,
         transforms.ToTensor(),
         #transforms.Normalize((0.5,)*d, (0.5,)*d),
         ts,
         sampleToBatch,
         toCuda,
         firstSection])
    
    # this is required to get activation of the reference model
    evalSet = torchvision.datasets.ImageFolder(evaluationFolder, loader=plt.imread, transform=transform )
    evalLoader = torch.utils.data.DataLoader(evalSet, batch_size=1, shuffle=False, num_workers=0)

    
    

    if TRAIN_CONCEPT:
        if USE_BRODEN:
            broden = brodenDataset('./datasets/broden1_224/', transform, resize=(7, 7))                
            conceptTrainLoader = data.DataLoader(broden.get_train_concept(concept_no), num_workers=0)
            conceptValidLoader = data.DataLoader(broden.get_valid_concept(concept_no), num_workers=0)
        else:
            conceptTrainLoader, conceptValidLoader, conceptTestLoader = loadDataset(conceptFolder, transform=transform, num_workers=0)
        
        #optimizer = optim.SGD(conceptModel.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(conceptModel.parameters())
                
        #train to retrieve information
        print(f'training for the layerNo = {layerNo}')
        logFile.write(f"conceptModel:{conceptModel.__class__.__name__}, layerNo:{layerNo}, optimizer: {optimizer.__class__.__name__}, N_EPOCHS:{N_EPOCHS}, smart_stop:{smart_stop}\n")
        trainForEpoches(conceptModel, conceptTrainLoader, conceptValidLoader, optimizer, criterion, N_EPOCHS = N_EPOCHS, smart_stop=smart_stop, resultFile=logFile, hardNeg=True)
        
        if savemodels:
            torch.save(conceptModel.state_dict(), savemodels+f'concept-{conceptModel.__class__.__name__}-layer{layerNo}.pt')
        
        
        print(f'ready for validation layerNo = {layerNo}')
        
        valid_loss, valid_acc = evaluate(conceptModel, conceptValidLoader, criterion)
        conceptValidLayersAcc.append(valid_acc)
        
        print('BCE loss:')
        print(f"valid_acc={valid_acc}")
        print(f"valid_loss={valid_loss}")
        logFile.write(f"BCE loss\n")
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
    conceptFilename = f"{logFilePath}-{layerNo}-concept.npy"
    targetFilename = f"{logFilePath}-{layerNo}-target.npy"
    
    logFile.write(f"writng concept to {conceptFilename}\n")
    np.save(conceptFilename, conceptMat)
    logFile.write(f"writng target to {targetFilename}\n")
    np.save(targetFilename, targetMat)
    logFile.flush()
    
    
    

print("conceptValidLayersAcc=", conceptValidLayersAcc)    
#plt.plot(conceptValidLayersAcc)
    
logFile.close()
