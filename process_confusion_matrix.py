#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:46:42 2018

@author: haoyang
"""

import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2

from multiprocessing import Pool

NUM_CLASSES = 101

data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

filename = './single_frame_confusion_matrix.npy'
#filename = './sequence_confusion_matrix.npy'
#filename = './combined_confusion_matrix.npy'
cm = np.load(filename)
# print(cm)

results = np.diag(cm)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

# print from the worst performance class to the best performance class
print('Performance of each class (from worst to best')
for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

# get the most confused classes
k_confused = 10
# set diagonal elements to 0
np.fill_diagonal(cm, 0)
print('Top 10 most confused classes')
for i in range(k_confused):
	idx = np.unravel_index(cm.argmax(), cm.shape)
	print(class_list[idx[0]],class_list[idx[1]])
	cm[idx] = 0

