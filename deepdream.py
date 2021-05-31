import torch
import torch.nn as nn
import torchvision.transforms
import numpy as np
from model import *
import matplotlib.pyplot as plt
from auxLearn.auxLearnVision import CUDA
from scipy.ndimage import rotate, zoom, shift
import progressbar
from random import randint
import scipy.ndimage as nd
import tqdm
import names

#adapted from https://github.com/eriklindernoren/PyTorch-Deep-Dream.git (MIT copyright)


#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
    
preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),# torchvision.transforms.Normalize(mean, std)
                                             ])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    #image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 1.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
    #    m, s = mean[c], std[c]
    #    image_tensor[0, c] = np.clip(image_tensor[0, c], -m / s, (1 - m) / s)
        image_tensor[0, c] = np.clip(image_tensor[0, c].cpu(), 0, 1)    
    return image_tensor


def dream(original_image, model, iterations, lr, target):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if torch.cuda.is_available():
        model = model.cuda()
    max_jitter = int(original_image.shape[-1]/8)
    max_rotate = 10
    
    print(original_image.shape)
    for i in tqdm.tqdm(range(iterations), desc='iterations'):
        #plt.imshow(deprocess(original_image))
        #plt.show()
        
        #rand_rotate = randint(-max_rotate, max_rotate + 1)
        #original_image = rotate(original_image, rand_rotate, reshape=False, axes=(-2,-1))
        
        shift_x, shift_y = randint(-max_jitter, max_jitter + 1), randint(-max_jitter, max_jitter + 1)
        original_image = np.roll(original_image, (shift_x,shift_y), (-1,-2))
        
        image = torch.autograd.Variable(Tensor(original_image), requires_grad=True)
        model.zero_grad()
        out = model(image)
        #loss = out.norm()
        #loss = torch.dot(out.view(-1), Tensor(target).view(-1))
        loss = out.view(-1)[target]
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        if avg_grad == 0:
            norm_lr = 0        
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        original_image = image.cpu().data.numpy()
        
        original_image = np.roll(original_image, (-shift_x,-shift_y), (-1,-2))
        
        #original_image = rotate(original_image, -rand_rotate,reshape=False, axes=(-2,-1))
        
        image.grad.data.zero_()
        
    plt.imshow(deprocess(original_image))
    plt.ion()
    plt.draw()
    plt.pause(.01)
    plt.show()
        
    return original_image


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves, target):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()
    

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr, target)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)

motherName = './models/resnet18_places365.pth.tar'
conceptName = './models/14.pth'

referenceModel = torchvision.models.__dict__['resnet18'](num_classes=365)
referenceModel.eval()

checkpoint = torch.load(motherName, map_location='cpu')
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
referenceModel.load_state_dict(state_dict)


conceptModel = SingleSigmoidFeatureClassifier()
conceptModel.load_state_dict(torch.load(conceptName))

L1 = getModules(referenceModel)[:-2]
model = nn.Sequential(nn.Sequential(*L1), conceptModel)

image = np.array(np.random.normal(0.5, .2, (256, 256, 3)), dtype='float')



for concept_ind in range(24):
    
    target = np.zeros((1,660))
    target[0,concept_ind] = 1
    
    
    dreamed_image = deep_dream(
        image,
        model,
        iterations=30,
        lr=0.01,
        octave_scale=1.4,
        num_octaves=7,
        target=concept_ind
    )
    
    plt.clf()
    plt.imshow(  dreamed_image  )
    plt.ioff()
    
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    
    plt.gcf().suptitle(names.concept_names[concept_ind])
    plt.pause(.01)
    plt.gcf().set_size_inches(3.5, 3.5)
    #plt.gcf().tight_layout()
    plt.savefig(f'dream_{concept_ind}_{names.concept_names[concept_ind]}.jpg')
    plt.close()