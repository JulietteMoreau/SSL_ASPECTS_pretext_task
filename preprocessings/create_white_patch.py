#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:03:56 2023

@author: moreau
"""

import nibabel as nib
import random as rd
import numpy as np
import os
from skimage.transform import rotate
import pandas as pd
import shutil as sh

patients = os.listdir('/path/to/input/raw/data/')

for pat in patients:
    im = nib.load('/path/to/input/raw/data/'+pat)
    image = im.get_fdata()
    
    array_min = image.min()
    array_max = image.max()
    
    valeur = np.max(image)+1
        
    # values for 192*192*90 images, adapt to the size of the image and the object in the image
    x = rd.randint(96, 110)
    y = rd.randint(40, 130)
    z = rd.randint(5, 80)
        
    l = rd.randint(x+20, min(x+60, 150))
    L = rd.randint(y+20, min(y+110, 160))
    p = rd.randint(z+5, min(z+60, 88))
        
    image[x:l, y:L, z:p] = valeur
        
    image = nib.Nifti1Image(image, im.affine, im.header)
    nib.save(image, '/path/to/output/data/'+pat)
    
