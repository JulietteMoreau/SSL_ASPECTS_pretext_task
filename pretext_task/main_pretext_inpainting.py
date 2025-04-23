#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:25:22 2022

@author: moreau
"""

import matplotlib.pyplot as plt
import os
import glob

# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torchsummary and torchvision
from torchsummary import summary

# numpy and pandas
import numpy as np

# Common python packages
import datetime
import os
import sys
import time

#monai stuff
from monai.transforms import LoadImageD, EnsureChannelFirstD, ToTensorD, Compose
from monai.data import CacheDataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#################################################################################################################################load data

dossier = sys.argv[1]
outdir = sys.argv[3]

if not os.path.exists(outdir):
    os.makedirs(outdir)   

KEYS = ("cerveau", "GT")

# load images
patients = os.listdir(dossier+'/image/train/')
images =[]
labels = []
for p in patients:
    if p[0]!='.':
        images.append(dossier+'/image/train/'+p)
        labels.append(dossier+'/ref/train/'+p)
        
patients = os.listdir(dossier+'/image/val/')
val_images =[]
val_labels = []
for p in patients:
    if p[0]!='.':
        val_images.append(dossier+'/image/val/'+p)
        val_labels.append(dossier+'/ref/val/'+p)

# set training pairs
train_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(images, labels)
]
val_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images, val_labels)
]

# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 2 # batch size, can be adapted to the machine capacities
train_ds = CacheDataset(data=train_files, transform=xform, num_workers=10)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=xform, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True)


################################################################################################################ parameters
# Hyperparameter for training
lr = float(sys.argv[2])

# Number of epochs and patience for early stopping
num_epoch = 200
patience = 10

# ################################################################################################################ generator Unet adapted for the inpainting
from inpaint_UNet import UNet
# Summary of the generator, adapt the size depending on the data
summary(UNet().cuda(), (1, 192, 192, 90))

# training
from train_pretext_inpainting import train_net
generator = train_net(train_loader, train_ds, val_loader, outdir, patience = patience, num_epoch=num_epoch, lr=lr)



