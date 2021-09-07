import datetime
import os.path

homedir = os.path.expanduser("~")
logFilePath = f'{homedir}/results-{datetime.datetime.now()}.txt'
logFilePath = logFilePath.replace(':','_').replace(' ','.')

# set to save cocnept model
savemodels = ''

inspectionLayers = [2, 5] # must be changed to 4 for BRODEN training

TRAIN_MODEL = False
TRAIN_CONCEPT = True
USE_BRODEN = False

# reference model path
motherName = './models/temp-best_5_acc72.pt'
# reference comcept model path
conceptName = ''

# if TRAIN_MODEL is True, set to the folder of training data
trainFolder = ''
dataFolder = './datasets/home/nokhbeh1/oai'
# if TRAIN_CONCEPT is True and USE_BRODEN is False, set to the folder of training concept data
conceptFolder = ''
# has to be set if USE_BRODEN and TRAIN_CONCEPT are both True
concept_no = 154

# folder of evaluation points (distribution sample set)
evaluationFolder = './datasets/places365/'

N_EPOCHS_TRAIN = N_EPOCHS = 500
# number of iternations that cost doesn't go down before stopping training
smart_stop = 50
# number of channels of input images
d = 3 
