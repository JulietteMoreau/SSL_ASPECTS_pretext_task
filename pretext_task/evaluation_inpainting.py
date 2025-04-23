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
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import nibabel as nib


def select_parts(image):
    # determination of the limits of the region to reconstruct, ie the region to evaluate
    x_min = 96
    x_max = 0
    y_min = 192
    y_max = 0
    z_min = 90
    z_max = 0
    for i in range(96, 192):
        for j in range(192):
            for k in range(90):
                if image[i,j,k]==numpy.max(image):
                    if i>x_max:
                        x_max=i
                    if j<y_min:
                        y_min=j
                    if j>y_max:
                        y_max =j
                    if k<z_min:
                        z_min=k
                    if k>z_max:
                        z_max=k
    return (x_min-1, x_max+1, y_min-1, y_max+1, z_min-1, z_max+1)

def evaluate_generator(generator, test_loader, test_files):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating Mask-w images

    """
    
    # initialization of a list for performance saving
    MSE= []
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    with torch.no_grad():

        # iterate over test set
        for i, batch in enumerate(test_loader):

            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = batch["GT"]
            real_Mask = (batch["GT"]>0)*batch["GT"]
            real_Mask = real_Mask.type(Tensor).int()
            
            max_value = torch.max(real_CT)-1
            
            fake_Mask = generator(real_CT, max_value)            
            

            real_CT = Tensor.cpu(real_CT).numpy()[0,0,:,:,:]
            real_Mask = Tensor.cpu(real_Mask).numpy()[0,0,:,:,:]
            fake_Mask = Tensor.cpu(fake_Mask).numpy()[0,0,:,:,:]

            # cut around the region of interest
            x_min, x_max, y_min, y_max, z_min, z_max = select_parts(real_CT)
            real_Mask_cut = real_Mask[x_min:x_max, y_min:y_max, z_min:z_max]
            fake_Mask_cut = fake_Mask[x_min:x_max, y_min:y_max, z_min:z_max]

            # calculate MSE
            MSE.append(mean_squared_error(real_Mask_cut.flatten(), fake_Mask_cut.flatten()))

    return MSE
        

KEYS = ("cerveau", "GT")


xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 1


############## test

from inpaint_UNet import UNet

# load data
dossier = "/path/to/data/"    

patients = os.listdir(dossier+'/image/')
images =[]
labels = []
images = sorted(glob.glob(dossier+'/image/val/'+'*.nii.gz'))
labels = sorted(glob.glob(dossier+'/ref/val/'+'*.nii.gz'))
    
KEYS = ("cerveau", "GT")

files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(images, labels)
]

    
print("Number test files: "+str(len(files)))

# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

test_ds = CacheDataset(data=files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# load model
generator = UNet()
generator.cuda()
generator.load_state_dict(torch.load('/path/to/weights/checkpoints/checkpoint.pth'))
generator.eval()
MSE = evaluate_generator(generator, test_loader, files)

# show results
print(" ")
print(lr, numpy.nanmean(MSE), numpy.nanstd(MSE))
print(" ")

