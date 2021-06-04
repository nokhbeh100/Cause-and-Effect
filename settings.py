import datetime
import os.path

homedir = os.path.expanduser("~")
logFilePath = f'{homedir}/results-{datetime.datetime.now()}.txt'
logFilePath = logFilePath.replace(':','_').replace(' ','.')

# set to save cocnept model
savemodels = ''

TRAIN_MODEL = False
TRAIN_CONCEPT = False
USE_BRODEN = False

# reference model path
motherName = './models/resnet18_places365.pth.tar'
# reference comcept model path
conceptName = './models/14.pth'

# if TRAIN_MODEL is True, set to the folder of training data
trainFolder = ''
# if TRAIN_CONCEPT is True and USE_BRODEN is False, set to the folder of training concept data
conceptFolder = ''
# has to be set if USE_BRODEN and TRAIN_CONCEPT are both True
concept_no = 0

# folder of evaluation points (distribution sample set)
evaluationFolder = './datasets/places365/'

N_EPOCHS_TRAIN = N_EPOCHS = 50
# number of iternations that cost doesn't go down before stopping training
smart_stop = 5
# number of channels of input images
d = 3 
