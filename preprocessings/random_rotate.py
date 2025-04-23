#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:42:17 2023

@author: moreau
"""

import nibabel as nib
import random as rd
import numpy as np
import os
from skimage.transform import rotate
import pandas as pd
import shutil as sh

rotation = []
pat = []

patients = os.listdir('/path/to/input/raw/images/')


for p in patients:
    rot = rd.randint(0,3)
    if rot!=0:
        im = nib.load('/path/to/input/raw/images/'+p)
        image = im.get_fdata()
        if rot==1:
            image = rotate(image, 90)
            image = nib.Nifti1Image(image, im.affine, im.header)
            nib.save(image, '/path/to/output/folder/'+p)
        elif rot==2:
            image = rotate(image, 180)
            image = nib.Nifti1Image(image, im.affine, im.header)
            nib.save(image, '/path/to/output/folder/'+p)
        elif rot==3:
            image = rotate(image, 270)
            image = nib.Nifti1Image(image, im.affine, im.header)
            nib.save(image, '/path/to/output/folder/'+p)
    pat.append(p)
    rotation.append(rot)        
            
    
    
zipped = list(zip(pat, rotation))

df = pd.DataFrame(zipped, columns=['patient', 'rotation'])
    
df.to_excel('/path/to/output/folder/rotation.xlsx')
