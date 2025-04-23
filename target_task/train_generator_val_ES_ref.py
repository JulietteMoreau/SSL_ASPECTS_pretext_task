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

################################################################################################################ dice_loss
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


############################################################################################################## validation step function
def evaluate(net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    
    dice_score = 0
    L = 0
    nb = 0
    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['cerveau'], batch['GT']
        mask_true = (mask_true>0)*1
        nb += image.shape[0]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            
            loss = criterion(mask_pred, mask_true.long().squeeze(1))+ dice_loss(F.softmax(mask_pred, dim=1).float(), F.one_hot(mask_true.long().squeeze(1), net.n_classes).permute(0,4,1,2,3).float(), multiclass=False)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, F.one_hot(mask_true.long().squeeze(1), net.n_classes).permute(0, 4, 1, 2, 3).float(), reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 4, 1, 2, 3).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], F.one_hot(mask_true.long().squeeze(1), net.n_classes).permute(0, 4, 1, 2, 3).float()[:, 1:, ...], reduce_batch_first=False)

            L += loss.item() 

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return L/nb, dice_score
    
    return L/nb, dice_score / num_val_batches


################################################################################################################train
def train_net(train_loader, train_ds, val_loader, outdir, patience=5, num_epoch=500,
               lr=0.0001, wd=0, delta = 3, amp=False):

    
    # initialization of the loss savings    
    loss_values=[]
    loss_val = []
    # initialization for the early-stopping
    trigger_times = 0
    last_loss = 100.0000
    
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


    # Initialize generator and discriminator
    net = UNet()
    
    # Optimizer and loss
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]))
    global_step = 0
    
    if cuda:
        net = net.cuda()
        criterion.cuda()

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    # loop over the number of epochs
    for epoch in range(num_epoch):
        
        running_loss = 0.0
               
        # loop over the train dataset
        for i, batch in enumerate(train_loader):

            # Inputs CT and Mask
            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = batch["GT"]
            real_Mask = (real_Mask>0)*1
            real_Mask = real_Mask.type(Tensor)

            # predict the segmentation mask
            fake_Mask = net(real_CT.float())
                       
            # calculate the loss
            loss = criterion(fake_Mask, real_Mask.long().squeeze(1)) + dice_loss(F.softmax(fake_Mask, dim=1).float(), F.one_hot(real_Mask.long().squeeze(1), net.n_classes).permute(0,4,1,2,3).float(), multiclass=False)

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

        # ----------
        #  Evaluation
        # ----------

        val_loss, val_score = evaluate(net, val_loader, device, criterion)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(val_score)

        sys.stdout.write(
            "\rValidation Dice score %f"
            % (val_loss))
        loss_values.append(running_loss / len(train_ds))
        loss_val.append(val_loss)
        

        # ----------
        #  Early-stopping
        # ----------        

        if limitcomma(val_loss, delta) >= limitcomma(last_loss,delta):
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            # stop training if there is no improvment
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                
                # save weights
                torch.save(net.state_dict(), outdir+"/checkpoints/checkpoint.pth".format(lr, wd, delta))
                logging.info(f'Checkpoint {epoch + 1} saved!')
                
                # plot loss
                plt.plot(loss_values)
                plt.plot(loss_val)
                plt.legend(['train', 'val'])
                plt.title("Loss d'apprentissage")
                plt.xlabel('Epoques')
                plt.savefig(outdir+"/images/loss_lr{}_wd{}_delta{}.png".format(lr, wd, delta))
            
                return net
            
        else:
            trigger_times = 0
            last_loss = val_loss

        
        # ----------
        # End training
        # ----------

        if epoch==num_epoch-1 or epoch==num_epoch:
            # save weights
            torch.save(net.state_dict(), outdir+"/checkpoints/checkpoint.pth".format(lr, wd, delta))
            logging.info(f'Checkpoint {epoch + 1} saved!')
            
    # plot loss
    plt.plot(loss_values)
    plt.plot(loss_val)
    plt.legend(['train', 'val'])
    plt.title("Loss d'apprentissage")
    plt.xlabel('Epoques')
    plt.savefig(outdir+"/images/loss_lr{}_wd{}_delta{}.png".format(lr, wd, delta))

    return net

