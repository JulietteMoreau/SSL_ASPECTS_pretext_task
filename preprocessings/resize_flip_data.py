#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:42:44 2024

@author: moreau
"""

import os
import numpy as np
import nibabel as nib
import skimage.transform as skTrans
from nibabel.affines import apply_affine
from nibabel.orientations import aff2axcodes
import matplotlib.pyplot as plt

# =============================================================================
# code to resize the data to the same size as the target data, and flip the unhealthy brain so that the lesions are in the same size
# =============================================================================

# list all the patients
patients= os.listdir('/path/to/images/')
patients.sort()

# iterate over the patients
for p in patients:
    
    # load image
    image = nib.load('/path/to/images/'+p)
    inter = image.get_fdata()
    
    # selective flip if it is an unhealthy brain 
    if p in os.listdir('/path/to/lesion/masks/'): # if there is a mask, it is unhealthy
    
        # load mask
        masque_im = nib.load('/path/to/lesion/masks/'+p)
        masque = masque_im.get_fdata()
        
        # see the size of the lesion
        if np.count_nonzero(masque[masque.shape[1]//2:,:,:]) < np.count_nonzero(masque[:masque.shape[1]//2,:,:]):
    
            # flip the mask and the brain
            inter = np.flip(inter, axis=0)
            masque = np.flip(masque, axis=0)

            
            sortie = nib.Nifti1Image(masque, masque_im.affine, masque_im.header)
            nib.save(sortie, '/path/to/lesion/masks/'+p)
            
            sortie = nib.Nifti1Image(inter, image.affine, image.header)
            nib.save(sortie, '/path/to/images/'+p)
    
    
    # resize image if it is not the same as wanted
    if inter.shape[0]!=192 or inter.shape[1]!=192 or inter.shape[2]!=90:
        
        # cases height and width are not the same: complete with black borders           
        if inter.shape[0]<inter.shape[1]:
            carre = np.zeros((inter.shape[1], inter.shape[1], inter.shape[2]))
            carre[(inter.shape[1]-inter.shape[0])//2:inter.shape[0]+(inter.shape[1]-inter.shape[0])//2,:,:] = inter.copy()
        elif inter.shape[0]>inter.shape[1]:
            carre = np.zeros((inter.shape[0], inter.shape[0], inter.shape[2]))
            carre[:,(inter.shape[0]-inter.shape[1])//2: inter.shape[1]+(inter.shape[0]-inter.shape[1])//2, :] = inter.copy()
        else:
            carre = inter.copy()
        
        # resize images
        sortie = skTrans.resize(carre, (192,192,90), order=1, preserve_range=True)
        # adapt size of the voxel in the header
        image.header["pixdim"][1] = image.shape[0]*image.header["pixdim"][1]/192
        image.header["pixdim"][2] = image.shape[1]*image.header["pixdim"][2]/192
        image.header["pixdim"][3] = image.shape[2]*image.header["pixdim"][3]/90
        
        
        sortie = nib.Nifti1Image(sortie, image.affine, image.header)
        nib.save(sortie, '/path/to/images/'+p)
        

        