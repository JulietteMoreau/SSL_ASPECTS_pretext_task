#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:04:37 2022

@author: moreau
"""


import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import roc_curve, auc

# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torchsummary and torchvision
from torchsummary import summary
from torchvision.utils import save_image

# matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.image as img

# numpy and pandas
import numpy
import pandas as pd
import random as rd
from PIL import Image

# Common python packages
import datetime
import os
import sys
import time

#monai stuff
from monai.transforms import RandSpatialCropSamplesD,SqueezeDimd, SplitChannelD,RandWeightedCropd,\
    LoadImageD, EnsureChannelFirstD, AddChannelD, ScaleIntensityD, ToTensorD, Compose, CropForegroundd,\
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandAffineD, CopyItemsd, OneOf, RandCoarseDropoutd, RandFlipd, NormalizeIntensityD
from monai.data import CacheDataset

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from monai.inferers import sliding_window_inference

import nibabel as nib


def dc(result, reference):


    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))

    intersection = numpy.count_nonzero(result & reference)

    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc



def evaluate_generator(generator, test_loader):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating Mask-w images

    """
    
    # initialize lists 
    dice= []
    dice_masse = [] #dice of the overall segmentation
    dices = [0,0,0,0,0,0,0,0,0,0] #save one value for each ASPECTS zone
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    with torch.no_grad():

        # iterate over the test set
        for i, batch in enumerate(test_loader):

            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = (batch["GT"]>0)*batch["GT"]
            real_Mask = real_Mask.type(Tensor)
            
            # predict ASPECTS zone
            fake_Mask = generator(real_CT)
            fake_Mask = torch.argmax(fake_Mask, dim=1).reshape((fake_Mask.shape[0],1,192,192, 90)) # adapt size according to your image size
                        
            real_CT = Tensor.cpu(real_CT).numpy()
            real_Mask = Tensor.cpu(real_Mask).numpy()
            fake_Mask = Tensor.cpu(fake_Mask).numpy()
            
            # dice of the overall zones            
            ref = (real_Mask>0)*1
            res = (fake_Mask>0)*1
            intersection = numpy.count_nonzero(res & ref)

            dice_masse.append(dc(res, ref))
            
            # dice of each zone
            somme_dice = 0
            for z in range(1,11):
                    
                result = (fake_Mask==z)

                reference = (real_Mask==z)
                intersection = numpy.count_nonzero(result & reference)
            
                size_i1 = numpy.count_nonzero(result)
                size_i2 = numpy.count_nonzero(reference)
                if size_i1!=0:
                    dce = 2. * intersection / float(size_i1 + size_i2)
                else:
                    dce=0
                    
                somme_dice+=dce
            
                dices[z-1]+=dce
            
            # mean of the 10 zines Dice
            dice.append(somme_dice/10)

    return dice, dices, dice_masse
        

KEYS = ("cerveau", "GT")


xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 1


############## test

from generator_UNet import UNet

# load data
dossier = '/path/to/test/data/'
patients = os.listdir(dossier+'ref/test/')
images =[]
labels = []
for p in patients:
    images.append(dossier+'image/test/'+p)
    labels.append(dossier+'ref/test/'+p)

KEYS = ("cerveau", "GT")

# constitute pairs
files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(images, labels)
]


test_ds = CacheDataset(data=files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

  
# load model
generator = UNet(n_classes=11)
generator.cuda()
generator.load_state_dict(torch.load('/path/to/checkpoints/checkpoint.pth'))
generator.eval()
d, dices, dice_masse = evaluate_generator(generator, test_loader)

# show results
print(" ")
print(numpy.nanmean(d), numpy.nanstd(d))

print(" ")
for i in range(len(dices)):
    print(dices[i]/len(files))
print()
print(numpy.nanmean(dice_masse))
