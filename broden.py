#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:26:37 2019

@author: mnz
"""

import torchvision
import torch
from torch.utils.data import Dataset
import pandas as pd
from matplotlib import pyplot as plt
import os
import names
import numpy as np
from auxLearn.auxLearnVision import resize

#import matplotlib._png as png
import cv2


def readMask(filename):
    #d = png.read_png_int(open(filename,'rb'))
    #d = plt.imread(filename)
    d = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    d = d.astype(np.uint32)
    return d[:,:,0] + 256*d[:,:,1] + 256*256*d[:,:,2]

class conceptDataset(Dataset):
    def __init__(self, parent, idx, concept_no, cat, train_size, resize=None):
        self.parent = parent
        self.idx = self.test_idx = np.where(idx)[0]
        self.train_idx = self.test_idx[:train_size]
        self.concept_no = concept_no
        self.cat = cat
        self.train_size = train_size
        self.resize = resize
        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, sampleNo):

        img_name = os.path.join(self.parent.images_dir, self.parent.index_frame['image'][self.idx[sampleNo]])
        image = plt.imread(img_name)

        if self.parent.transform:
            image = self.parent.transform(image)
        
        mask_names = self.parent.index_frame[self.cat][self.idx[sampleNo]]
        out = 0 
        for mask_name in mask_names.split(';'):
            mask_name = os.path.join(self.parent.images_dir, mask_name)
            mask = readMask(mask_name)
            # if output is desired to be image, remove the mean
            if self.resize:
                out += resize(1000.*(mask == self.concept_no), size=self.resize)/1000.
            else:
                out += np.mean(mask == self.concept_no)

        return image, torchvision.transforms.ToTensor()(out).to(torch.float32)
    
    def apply_hardNeg(self, errVal):
        self.train_idx = self.test_idx[ np.argsort(errVal)[-self.train_size:] ]
        
    def eval(self):
        self.idx = self.test_idx
        
    def train(self):
        self.idx = self.train_idx
        

    
class brodenDataset:
    """Broden interface."""

    def __init__(self, root_dir, transform=None, size_factor=20, resize=None):#, cats = ['object', 'part']):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir,'images')
        self.index_frame = pd.read_csv(os.path.join(root_dir,'index.csv'))
        self.label_frame = pd.read_csv(os.path.join(root_dir,'label.csv'))
        
        self.transform = transform
        self.size_factor = size_factor
        self.resize = resize
            
    def get_cat(self, concept_no):
        idx = self.label_frame['number'] == concept_no
        text = self.label_frame['category'][idx].values[0]
        for cat in names.CATAGORIES:
            if cat in text:
                break
            
        pos_size = self.label_frame['frequency'][idx].values[0]
        return cat, pos_size
    
    def get_train_concept(self, concept_no):
        cat, pos_size = self.get_cat(concept_no)
        indexes = ~(pd.isna(self.index_frame[cat])) & (self.index_frame['split'] == 'train')
        return conceptDataset(self, indexes, concept_no, cat, int(pos_size * self.size_factor), resize=self.resize)
    
    def get_valid_concept(self, concept_no):
        cat, pos_size = self.get_cat(concept_no)
        indexes = ~(pd.isna(self.index_frame[cat])) & (self.index_frame['split'] == 'val')
        return conceptDataset(self, indexes, concept_no, cat, int(pos_size * self.size_factor), resize=self.resize)
