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
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandAffineD, CopyItemsd, OneOf, RandCoarseDropoutd, RandFlipd
from monai.data import CacheDataset

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure

import nibabel as nib

def truncate(n):
    return int(n * 100)/100


def dc(result, reference):
    # measure dice
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


def hd(result, reference, voxelspacing=None, connectivity=1):
    # measure hausdorff distance
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def ravd(result, reference):
    #measure ravd
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))

    vol1 = numpy.count_nonzero(result)
    vol2 = numpy.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return (vol1 - vol2) / float(vol2)

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == numpy.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the inumpyut has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


############################################################################################## evaluation function

def evaluate_generator(generator, train_loader, test_files, t):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating Mask-w images

    """
    
    # initialize lists for saving
    res_train, res_test = [], []
    compte = 0
    dice= []
    HD = []
    RAVD = []

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    with torch.no_grad():

        # iterate over test set
        for i, batch in enumerate(test_loader):
            
            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = (batch["GT"]>0)*1
            real_Mask = real_Mask.type(Tensor)
            fake_Mask = generator(real_CT)
            fake_Mask = torch.argmax(fake_Mask, dim=1).reshape((fake_Mask.shape[0],1,192,192, 90))
            
            
            real_CT = Tensor.cpu(real_CT).numpy()
            real_Mask = Tensor.cpu(real_Mask).numpy()
            fake_Mask = Tensor.cpu(fake_Mask).numpy()

            # values if the prediction is empty
            if numpy.count_nonzero(fake_Mask)==0:
                HD.append(56) # around half of the brain height
                dice.append(0)
                RAVD.append(-1)
                compte +=1

            else:
                dice.append(dc(fake_Mask, real_Mask))
                HD.append(hd(fake_Mask,real_Mask))
                RAVD.append(ravd(fake_Mask,real_Mask))

            res_test.append([dice[-1], HD[-1], RAVD[-1]])
    
            
        df = pd.DataFrame([
            pd.DataFrame(res_train, columns=['DICE', 'HD', 'RAVD']).mean().squeeze(),
            pd.DataFrame(res_test, columns=['DICE', 'HD', 'RAVD']).mean().squeeze()
            ], index=['Training set', 'Test set']).T
    return dice, HD, RAVD, df
        


KEYS = ("cerveau", "GT")


xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 1


############## test


dossier = "/path/to/test/data/"    


patients = os.listdir(dossier+'/image/test/')
images =[]
labels = []
for p in patients:
    images.append((dossier+'/image/test/'+p))
    labels.append((dossier+'/ref/test/'+p))


KEYS = ("cerveau", "GT")


# import the model architecture
from generator_UNet import UNet


# list of all pretext tasks 
tache = ['reference', 'rotation', 'inpainting', 'ASPECTS']
   

# load images
for i in range(len(images)):
    
    files = [
    {"cerveau": images[i], "GT": labels[i]}]
    

test_ds = CacheDataset(data=files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# evaluation over the different models
for t in tache:

    generator = UNet()
    generator.cuda()
    generator.load_state_dict(torch.load('/path/to/checkpoints/'+t+'/checkpoint.pth'))
    generator.eval()
    D, H, R, df = evaluate_generator(generator, test_loader, files, t)
    
            
    print(" ")
    print(numpy.nanmean(D), numpy.nanstd(D))
    print(numpy.nanmean(H), numpy.nanstd(H))
    print(numpy.nanmean(R), numpy.nanstd(R))
    print(" ")
    
        


