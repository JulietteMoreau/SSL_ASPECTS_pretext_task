#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:09:40 2023

@author: moreau
"""

import os
import numpy as np
import nibabel as nib
from skimage.morphology import disk, binary_closing
from skimage.measure import label, regionprops
import fsl.utils.image.resample as fsl
from fsl.data.image import Image, Nifti


zones = {'C_':1,'IC_':2,'I_':3,'L_':4,'M1_':5,'M2_':6,'M3_':7,'M4_':8,'M5_':9,'M6_':10}

patients_tot = os.listdir('/home/moreau/Documents/Segmentation/SSL/data/pretexte/ASPECTS/image/')
patients_tot.sort()

patients = []

for p in patients_tot:
    if p not in os.listdir('/home/moreau/Documents/Segmentation/SSL/data/pretexte/ASPECTS/ref'):
        patients.append(p)



for p in patients:
    
    print(p)
    p=p[:-7]

    ref = nib.load(os.path.join('/home/moreau/Documents/Segmentation/SSL/data/pretexte/ASPECTS/image/'+p+'.nii.gz'))
    ref_im = ref.get_fdata()
    ref_im = Image(ref_im)
    
    # ref_im = fsl.resample(ref_im, (192,192,90))
    # ref_im = ref_im[0]
    
    ASPECTS = np.zeros(ref_im.shape)
    
    for cote in ['R']:
    
        for z in range(len(zones)):
            
            zone = nib.load(os.path.join('/home/moreau/Documents/Segmentation/SSL/data/pretexte/ASPECTS/ref/', p, list(zones.keys())[z]+cote+'.nii.gz'))
            matrice = zone.get_fdata()
            #matrice = Image(matrice)
            
            # matrice = fsl.resample(matrice, (192,192,90))
            # matrice = matrice[0]
            
            mini = max(0, list(zones.values())[0]-0.4)
            maxi = list(zones.values())[0]+0.4
            
            matrice = ((matrice>mini) & (matrice<maxi))*1
            
            label_zone = label(matrice, connectivity=1)
            regions = regionprops(label_zone)
            A = np.max([r.area for r in regions])
            for r in regions:
                if r.area < A: # Aire trop petite
                    for coords in r.coords:  # Tous les pixels de la région passent à zéros              
                        label_zone[coords[0], coords[1], coords[2]] = 0
    
            label_zone = (label_zone>0)*1
    
            selem = disk(5)
            
            closed = np.zeros(ref_im.shape)
            for c in range(ref_im.shape[2]):
                coupe = label_zone[...,c]
                coupe_fermee = binary_closing(coupe, selem)
                closed[...,c] = coupe_fermee
                
            matrice = (closed>0)*(z+1)
            
            intersection = (ASPECTS>0) & (matrice>0)
            
            if np.count_nonzero(intersection)==0:
                ASPECTS = ASPECTS+matrice
            
            else:
                for i in range(ref_im.shape[0]):
                    for j in range(ref_im.shape[1]):
                        for k in range(ref_im.shape[2]):
                            if intersection[i,j,k]!=0:
                                ASPECTS[i,j,k]=0
                ASPECTS = ASPECTS + matrice
    
    # os.mkdir('/home/moreau/Documents/Segmentation/self_supervised/data/IXI_isles/'+p)

    out = nib.Nifti1Image(ASPECTS, ref.affine, ref.header)
    nib.save(out, os.path.join('/home/moreau/Documents/Segmentation/SSL/data/pretexte/ASPECTS/ref/', p+'.nii.gz'))
    
    # out = nib.Nifti1Image(ref_im, ref.affine, ref.header)
    # nib.save(out, os.path.join('/home/moreau/Documents/manuscrit/data/SSL/'+d+'_fin', 'ref',p, 'Relate_'+p+'_Ct_J1_.nii.gz'))
