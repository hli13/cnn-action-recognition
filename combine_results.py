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

##### save predictions directory
prediction_directory_1 = 'UCF-101-predictions/'
prediction_directory_2 = 'UCF-101-predictions_sequence/'

confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)

# loop over test data
for i in range(len(test[0])):

    t1 = time.time()

    index = i

    # choose a test video
    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')
    filename1 = filename.replace(data_directory+'UCF-101-hdf5/',prediction_directory_1)
    filename2 = filename.replace(data_directory+'UCF-101-hdf5/',prediction_directory_2)

    # read saved results
    # single frame model, one prediction vector for each frame
    h1 = h5py.File(filename1,'r')
    pred1 = np.array(h1['predictions'])
    nFrames = pred1.shape[0]

    # sequence model, one prediction vector for each video
    h2 = h5py.File(filename2,'r')
    pred2 = np.array(h2['predictions'])

    pred_comb = np.zeros((nFrames,NUM_CLASSES),dtype=np.float32)

    # softmax and combine
    pred2 = np.exp(pred2)/np.sum(np.exp(pred2))
    for j in range(pred1.shape[0]):
        pred1[j] = np.exp(pred1[j])/np.sum(np.exp(pred1[j]))
        pred_comb[j] = pred1[j] + pred2

    pred_comb = np.sum(np.log(pred_comb),axis=0)
    argsort_pred = np.argsort(-pred_comb)[0:10]

    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))

number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

np.save('combined_confusion_matrix.npy',confusion_matrix)