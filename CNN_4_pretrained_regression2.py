# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 04:47:03 2019

@author: aijing
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import cv2
from numpy import moveaxis
#import ipdb
"""
data =pd.read_csv('labelsForMeter.csv')

X=data.loc[:, ['name']]
Y=data.loc[:,'labels']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=0)

def nameToImage(x_name,image_size=224):
    m=x_name.shape[0]
    x_image=np.zeros((m,3,image_size,image_size)).astype(np.uint8)
    
    for i in range(m):
        path='E:/spyder/meter/'+str(x_name.iloc[i,0])
        I = cv2.imread(path)
        # ipdb.set_trace()
        pic = cv2.resize(I, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        x_image[i,:,:,:]=moveaxis(pic,2, 0)
        
    return x_image

x_train_image=nameToImage(x_train)
x_test_image=nameToImage(x_test)

y_train=y_train.values
y_test=y_test.values

def check_image(x_train_image,j):
    for i in range(j):
        path='E:/spyder/image_check/'+str(i)+'.tiff'
        img=x_train_image[i,:,:,:]
        img=np.uint8(moveaxis(img,0, 2))
        cv2.imwrite(path,img)
        
check_image(x_test_image,2)

"""
""""
x_train_image=np.load('x_train_image.npy')
y_train=np.load('y_train.npy').reshape([-1,1])
x_test_image=np.load('x_test_image.npy')
y_test=np.load('y_test.npy').reshape([-1,1])



"""
"""

"""
MINIBATCH_SIZE = 8
"""
train_inputs, train_labels = torch.FloatTensor(normalizedImage(x_train_image)), torch.FloatTensor(y_train[:])
test_inputs, test_labels = torch.FloatTensor(normalizedImage(x_test_image)), torch.FloatTensor(y_test[:])

loader_train = Data.DataLoader(dataset=Data.TensorDataset(train_inputs, train_labels),
                         batch_size=MINIBATCH_SIZE,shuffle=True,
                         num_workers=0)

loader_test = Data.DataLoader(dataset=Data.TensorDataset(test_inputs, test_labels),
                         batch_size=MINIBATCH_SIZE,shuffle=True,
                         num_workers=0)
dataloaders_dict={'train':loader_train,'val':loader_test}
a,b=next(iter(dataloaders_dict['train']))


del x_train_image, y_train,x_test_image,y_test
"""
model_name = "resnet18"

num_classes = 2
num_epochs = 20
feature_extract = False


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name=='resnet50':
        model_ft=models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name=='vgg16':
        model_ft=models.vgg16(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
"""
outputs = model_ft(torch.FloatTensor(normalizedImage(x_test_image)))
optimizer_ft.zero_grad()
loss = criterion(outputs, torch.FloatTensor(y_test[3:5]))
loss.backward()
optimizer_ft.step()
"""
# Print the model we just instantiated
"""
print(model_ft)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# criterion = nn.CrossEntropyLoss()
criterion =nn.L1Loss()
criterion_2 =nn.L1Loss(reduction='none')

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0

            # Iterate over data.
            for inputs, labels, area in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.float().to(device).view(-1,1)
                area = area.float().to(device).view(-1,1)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs= model(inputs)
                        label_and_area=torch.cat((labels,0.05*area), 1)
                        loss = criterion(outputs, label_and_area)
                        acc = torch.sum(1-criterion_2(outputs, label_and_area)/label_and_area)
                        

                    #_, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_acc += acc.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / (2*len(dataloaders[phase].dataset))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

#torch.save(model_ft.state_dict(),'resnet18_20_add_row4.pth')
# del net  
#torch.cuda.empty_cache()
"""