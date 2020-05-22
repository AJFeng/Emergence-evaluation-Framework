
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:13:24 2019

@author: af3bd
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
import torch.optim as optim
import cv2
import torchvision.transforms as tvt
from PIL import Image
import torchvision
from numpy import moveaxis
import torch.utils.data as Data
from CNN_4_pretrained_regression2 import initialize_model
import time
import sys
#import ipdb
#import psutil


#psutil.cpu_percent()
#psutil.virtual_memory
#dict(psutil.virtual_memory()._asdict())

model_name = "resnet18"
feature_extract = False
num_classes = 2
MINIBATCH_SIZE = 8

def normalizedImage(x,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    x=x/255
    
    for i in range(len(mean)):
        x[:,i,:,:]=(x[:,i,:,:]-mean[i])/std[i]
    
    return x

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
model_ft.cuda()
model_ft.load_state_dict(torch.load(sys.argv[3],map_location='cuda'))
model_ft.eval()

"""
x_test_image=np.load('x_test_image_224_meter.npy')
y_test=np.load('y_test_224_meter.npy').reshape([-1,1])
test_inputs, test_labels=torch.FloatTensor(normalizedImage(x_test_image)), torch.FloatTensor(y_test[:])
loader_test = Data.DataLoader(dataset=Data.TensorDataset(test_inputs, test_labels),
                         batch_size=MINIBATCH_SIZE,shuffle=True,
                         num_workers=0)

criterion_2 =nn.L1Loss(reduction='none')

running_acc=0
for inputs, labels in loader_test:                
    outputs = model_ft(inputs)
    loss = criterion_2(outputs, labels)
    acc = torch.sum(1-criterion_2(outputs, labels)/labels)
    running_acc += acc.item()

epoch_acc = running_acc / len(loader_test.dataset)
"""

#img_path='E:/matlab/0531phantom/segment/'

img_path=sys.argv[1]
#fName =pd.read_csv('gps.csv')
fName=pd.read_csv(sys.argv[2])
#ipdb.set_trace()

img_id=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
since = time.time()
x_image=np.zeros((MINIBATCH_SIZE,3,input_size,input_size)).astype(np.uint8)
seedling=torch.FloatTensor(np.zeros((len(fName),2)))
seedling.cuda()
while (img_id+7)<len(fName):
    for i in range(MINIBATCH_SIZE):
        path=img_path+str(fName.iloc[img_id+i,2])
        I = cv2.imread(path)
        pic = cv2.resize(I, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        x_image[i,:,:,:]=moveaxis(pic,2, 0)
    
    inputs=torch.FloatTensor(normalizedImage(x_image))
    inputs = inputs.to(device)
    outputs = model_ft(inputs)
    seedling[img_id:img_id+8,:]=outputs.detach()
    img_id=img_id+8

# the last batch that less than 8 data
x_image=np.zeros((len(fName)-img_id,3,input_size,input_size)).astype(np.uint8)
for i in range(len(x_image)):
    path=img_path+str(fName.iloc[img_id+i,2])
    I = cv2.imread(path)
    pic = cv2.resize(I, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    x_image[i,:,:,:]=moveaxis(pic,2, 0)
    
inputs=torch.FloatTensor(normalizedImage(x_image))
inputs = inputs.to('cuda')
outputs = model_ft(inputs)
seedling[img_id:len(fName),:]=outputs.detach()


time_elapsed = time.time() - since 

seedling_cpu=seedling.numpy()
#ipdb.set_trace()
#seedling_cpu=np.hstack((fName.iloc[:,0:2].values,seedling_cpu))

fName['stand_count']=seedling_cpu[:,0]
fName['overall_canopy_size']=seedling_cpu[:,1]/0.05
fName['canopy_size (cm2/seedlings)']=seedling_cpu[:,1]/0.05/seedling_cpu[:,0]
fName.to_csv("seedling.csv",index=False)
#np.savetxt("seedling.csv", seedling_cpu, delimiter=",")
