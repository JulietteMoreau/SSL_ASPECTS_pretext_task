#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:04:37 2022

@author: moreau
"""


import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

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



def evaluate_generator(generator, train_loader, test_files):
    """Evaluate a generator.

    Args:
        generator: (GeneratorUNet) neural network generating Mask-w images

    """


    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    with torch.no_grad():

        # initialize empty lists for results
        y_test = []
        y_pred = []
      
        # iterate over the test set
        for i, batch in enumerate(test_loader):

            # Inumpyuts CT-w and Mask-w
            real_CT = batch["cerveau"].type(Tensor)
            real_class = (batch["ASPECTS"]).type(Tensor)
            
            # predict rotation
            fake_class = generator(real_CT)

            fake_class = torch.argmax(fake_class)

            real_CT = Tensor.cpu(real_CT).numpy()
            real_class = Tensor.cpu(real_class).numpy()
            fake_class = Tensor.cpu(fake_class).numpy()

            y_test.append(real_class)
            y_pred.append(fake_class)

        # measure accuracy and confusion matrix
        acc = accuracy_score(y_test, y_pred)
        CM = confusion_matrix(y_test, y_pred)
    return acc, CM

KEYS = ("cerveau", "ASPECTS")


xform = Compose([LoadImageD(('cerveau')),
    EnsureChannelFirstD(('cerveau')),
    ToTensorD(KEYS)])

bs = 1

# load label table
table = pd.read_excel('/path/to/label/table/rotation.xlsx', engine='openpyxl')

############## test

# load architecture
from classifier_UNet import UNet

        
test_dir = '/path/to/test/data/'


# load images
test_im = sorted(glob.glob(test_dir + "*.nii.gz"))

test_labels = []
test_images = []
for CT in test_im:
    if CT.split('/')[-1] in table['patient'].values:
        test_labels.append(table.loc[table['patient'] == (CT.split('/')[-1]),'rotation'].values[0])
        test_images.append(CT)

# create data pairs
test_files = []
for im in range(len(test_images)):
    test_files.append({"cerveau": test_images[im], "ASPECTS": test_labels[im]})

test_ds = CacheDataset(data=test_files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)


# load model
generator = UNet(n_classes=4)
generator.cuda()
generator.load_state_dict(torch.load('/path/to/weights/checkpoints/checkpoint.pth'))
generator.eval()
# evaluate
acc, CM = evaluate_generator(generator, test_loader, test_files)

# show results
print(' ')
print(acc)
print(' ')
print(CM)
print('')

plt.imshow(CM)
plt.show()
