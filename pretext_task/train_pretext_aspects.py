#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:44:24 2022

@author: moreau
"""

import argparse
import logging
import sys
from pathlib import Path
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from torch import Tensor
from generator_UNet import UNet
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy

################################################################################################################dice_loss
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def limitcomma(value, limit=2):
     #truncature of float numbers
  v = str(value).split(".")
  return float(v[0]+"."+v[1][:limit])



############################################################################################################## validation function
def evaluate(net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    
    L = 0
    nb = 0
    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, true_image = batch['cerveau'], batch['GT']
        nb += image.shape[0]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_image = true_image.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            fake_image = net(image.float())
            
            loss = criterion(fake_image, true_image.long())
            
            L += loss.item() #* image.size(0)
           
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return L/nb
    
    return L/nb


################################################################################################################train
def train_net(train_loader, train_ds, val_loader, outdir, num_epoch=500,
               lr=0.0001, amp=False):

    # initialize loss lists
    loss_values=[]
    loss_val = []
    # initialize early stopping
    trigger_times = 0
    last_loss = 100.0

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Output folder
    if not os.path.exists(outdir+"/images"):
        os.makedirs(outdir+"/images")
        
    if not os.path.exists(outdir+"/checkpoints"):
        os.makedirs(outdir+"/checkpoints")  


    # Initialize model with 10 classes
    net = UNet(n_classes=11)

    # Optimizers
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0
    
    if cuda:
        net = net.cuda()
        criterion.cuda()

    
    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    # loop around the number of epochs
    for epoch in range(num_epoch):
        
        running_loss = 0.0
               
        #iterate over the training set
        for i, batch in enumerate(train_loader):

            # Inputs CT and Mask
            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = batch["GT"]
            
            real_Mask = real_Mask.type(Tensor)
            
            # predict masks
            fake_Mask = net(real_CT.float())
                       
            
            # loss
            loss = criterion(fake_Mask, real_Mask.long().squeeze(1))+ dice_loss(F.softmax(fake_Mask, dim=1).float(), F.one_hot(real_Mask.long().squeeze(1), net.n_classes).permute(0,4,1,2,3).float(), multiclass=True)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()


            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = num_epoch * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] "
                "ETA: %s"
                % (
                    epoch + 1,
                    num_epoch,
                    i,
                    len(train_loader),
                    grad_scaler.scale(loss).item(),
                    time_left,
                )
            )
            
            running_loss += loss.item() * real_CT.size(0) 


        loss_values.append(running_loss / len(train_ds))
        
        # ----------
        #  Validation
        # ----------
        
        val_loss = evaluate(net, val_loader, device, criterion)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(loss)

        sys.stdout.write(
            "\rValidation Dice score %f"
            % (val_loss))
        loss_val.append(val_loss)
        
        
        # ----------
        #  Early stopping
        # ----------
        
        # check loss evolution
        if limitcomma(val_loss, 3) >= limitcomma(last_loss,3):
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                
                # save weights
                torch.save(net.state_dict(), outdir+"/checkpoints/checkpoint.pth")
                logging.info(f'Checkpoint {epoch + 1} saved!')
                
                # plot loss
                plt.plot(loss_values)
                plt.plot(loss_val)
                plt.legend(['train', 'val'])
                plt.title("Loss d'apprentissage")
                plt.xlabel('Epoques')
                plt.savefig(outdir+"/images/loss.png")
            
                return net
            
        else:
            trigger_times = 0
            last_loss = val_loss
        
    # ----------
    #  End of training
    # ----------
        
    if epoch==num_epoch-1:
        # save weights
        torch.save(net.state_dict(), outdir+"/checkpoints/checkpoint.pth")
        logging.info(f'Checkpoint {epoch + 1} saved!')
            

    plt.plot(loss_values)
    plt.plot(loss_val)
    plt.legend(['train', 'val'])
    plt.title("Loss d'apprentissage")
    plt.xlabel('Epoques')
    plt.savefig(outdir+"/images/loss.png")

    return net