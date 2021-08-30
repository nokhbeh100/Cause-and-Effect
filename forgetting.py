
# cp ~/best-mothernet-vgg16-69-71-cropped-seeded.pt ./models/

from settings import *
from oai import *

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


transform = transforms.Compose(
    [gray2rgb,
     transforms.ToTensor(),
     lambda image:(image - torch.mean(image))/torch.std(image)])

dataSet = OAI(dataFolder, transform=transform)
trainSet, testSet, validSet = trainTestValid( dataSet, .7, .2, .1, seed=2021)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=16,
                                         shuffle=False, num_workers=2)

testLoader = torch.utils.data.DataLoader(testSet, batch_size=16,
                                         shuffle=False, num_workers=2)

validLoader = torch.utils.data.DataLoader(validSet, batch_size=16,
                                         shuffle=False, num_workers=2)


#%%

# both models must be converted to sequential models for better spliting points
#referenceModelBase = torchvision.models.__dict__['resnet18'](num_classes=365)
referenceModelBase = singleOutput(torchvision.models.resnet18(pretrained=True))
#referenceModelBase = singleOutput(torchvision.models.vgg16(pretrained=True))
# consider the fact that the model is coppied to local machine
torch.save(referenceModelBase.state_dict(), motherName)
referenceModel = nn.Sequential( *getModules(referenceModelBase.model) )


#motherNetBase = torchvision.models.__dict__['resnet18'](num_classes=365)
motherNetBase = singleOutput(torchvision.models.resnet18(pretrained=True))
#motherNetBase = singleOutput(torchvision.models.vgg16(pretrained=False))
motherNet = nn.Sequential( *getModules(motherNetBase.model) )

if CUDA:
    motherNet = motherNet.cuda()
    referenceModel = referenceModel.cuda()


criterion = nn.MSELoss()


# if model is pretrained
#checkpoint = torch.load(motherName, map_location='cpu')
#state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
state_dict = torch.load(motherName)

referenceModelBase.load_state_dict(state_dict)
referenceModel = nn.Sequential( *getModules(referenceModelBase.model) )

motherNetBase.load_state_dict(state_dict)
motherNet = nn.Sequential( *getModules(motherNetBase.model) )


#%%


print('Inspecting layers')

#labels = open(labels_path).read().splitlines()
#trying to build up from last to first

conceptValidLayersAcc = []
conceptValidLayersSen = []
layers = list( motherNet.children() )


for layerNo in inspectionLayers:

    print (f'inspecting: {-layerNo}')
    print(f'getting ready training for the layerNo = {layerNo}')


    if len(list(layers[-layerNo+1].parameters())) == 0:
        print('WARNING: next layer is nontrainable (maybe activation/dropout/flatten)')
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
         lambda image:(image - torch.mean(image))/torch.std(image),
         #transforms.Normalize((0.5,)*d, (0.5,)*d),
         #ts,
         sampleToBatch,
         toCuda,
         firstSection,
         batchToSample])


    dataSet = OAI(dataFolder, transform=transform)
    trainSet, testSet, validSet = trainTestValid( dataSet, .7, .2, .1, seed=2021)

    conceptTrainLoader = torch.utils.data.DataLoader(cacheDataset(trainSet), batch_size=16,
                                             shuffle=False, num_workers=0)

    conceptValidLoader = torch.utils.data.DataLoader(cacheDataset(testSet), batch_size=16,
                                             shuffle=False, num_workers=0)
    # this is required to get activation of the reference model
    #evalSet = torchvision.datasets.ImageFolder(evaluationFolder, loader=plt.imread, transform=transform )
    evalLoader = torch.utils.data.DataLoader(validSet, batch_size=1, shuffle=False, num_workers=0)




    if TRAIN_CONCEPT:

        optimizer = optim.SGD(conceptModel.parameters(), lr=0.001)
        #optimizer = optim.Adam(conceptModel.parameters())

        #train to retrieve information
        print(f'training for the layerNo = {layerNo}')
        logFile.write(f"conceptModel:{conceptModel.__class__.__name__}, layerNo:{layerNo}, optimizer: {optimizer.__class__.__name__}, N_EPOCHS:{N_EPOCHS}, smart_stop:{smart_stop}\n")
        trainForEpoches(conceptModel, conceptTrainLoader, conceptValidLoader, optimizer, criterion, N_EPOCHS = N_EPOCHS, smart_stop=smart_stop, resultFile=logFile, hardNeg=False)

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
