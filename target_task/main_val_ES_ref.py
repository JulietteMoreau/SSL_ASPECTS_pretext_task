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
from monai.transforms import LoadImageD, EnsureChannelFirstD, ToTensorD, Compose, NormalizeIntensityD
from monai.data import CacheDataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#################################################################################################################################load data

# getting the directories information
dossier = sys.argv[1]
outdir = sys.argv[2]

# loading train and validation images
patients = os.listdir(dossier+'/image/train/')
images = []
labels = []
for p in patients:
    images.append(dossier+'/image/train/'+p)
    labels.append(dossier+'/ref/train/'+p)
    
patients = os.listdir(dossier+'/image/validation/')
val_images =[]
val_labels = []
for p in patients:
    val_images.append(dossier+'/image/validation/'+p)
    val_labels.append(dossier+'/ref/validation/'+p)
        
if not os.path.exists(outdir):
    os.makedirs(outdir) 

KEYS = ("cerveau", "GT")

#link images and references
files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(images, labels)
]

val_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images, val_labels)
]


print("Number Train files: "+str(len(files)))
print("Number val files: "+str(len(val_files)))
# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 2 # batch size, can be adapted depending on the computer
train_ds = CacheDataset(data=files, transform=xform, num_workers=10)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=xform, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True)


################################################################################################################paramètres d'apprentissage
# Hyperarameters for training
lr = float(sys.argv[3])
wd = float(sys.argv[4])
delta = int(sys.argv[5])


# Number of epochs and patience for early-stopping
num_epoch = 150
patience = 5

# ################################################################################################################generator Unet
from generator_UNet import UNet
# Summary of the generator, adapt the size to your images size
summary(UNet().cuda(), (1, 192, 192, 90))

# launch training
from train_generator_val_ES_ref import train_net
generator = train_net(train_loader, train_ds, val_loader, outdir, patience=patience, num_epoch=num_epoch, lr=lr, wd=wd, delta=delta)

