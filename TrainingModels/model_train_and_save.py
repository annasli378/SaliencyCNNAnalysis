# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:14:41 2023

@author: annas
"""

# IMPORTY:
from __future__ import print_function, division

import torch
#import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

#from PIL import Image
import os

import numpy as np
import matplotlib.pyplot as plt
import time
import copy

import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import AlexNet_Weights#, ResNet101_Weights, DenseNet201_Weights, GoogLeNet_Weights,Inception_V3_Weights, SqueezeNet1_1_Weights, VGG19_Weights

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

if __name__ == '__main__':
    cudnn.benchmark = True
    plt.ion()   # interactive mode

    #%% ZMIENNE
    train_acc = []
    train_loss = []
   
    # path_to_test_dataset = "D:/Dataset_BUSI_with_GT/all/resized_k5_randomized"
    
    path_k1 = "D:/dataset_for_kfold5/k1"
    path_k2 = "D:/dataset_for_kfold5/k2"
    path_k3 = "D:/dataset_for_kfold5/k3"
    path_k4 = "D:/dataset_for_kfold5/k4"
    path_k5 = "D:/dataset_for_kfold5/k5"
    
    path_save_models = "D:/Testy_metod/nauczone_modele_kfold/alexnet"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = 4
    
# %% FUNKCJE
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        
    
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
    
            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
      
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'train':
                    train_acc.append(epoch_acc)
                    train_loss.append(epoch_loss)
                
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    def evaluate_trained_model(dataloaders, model):
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataloaders['test']:
                images, labels = data
                #print(labels)
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predicted.append(predicted)
                all_labels.append(labels)
        
        print(f'Accuracy of the network on the 126 test images: {100 * correct // total} %')
        return all_labels, all_predicted
    
    def tensors_to_list(all_labels, all_predicted):
        labels_list = []
        predicted_list = []
        if type(all_labels[0]) == torch.Tensor:
            # print('Zamiana listy tensorów na listę intów:')
            for tens in all_labels:
                tens_len = len(tens)
                for i in range(0,tens_len):
                    labels_list.append(tens[i].item())
            for tens in all_predicted:
                tens_len = len(tens)
                for i in range(0,tens_len):
                    predicted_list.append(tens[i].item())
            print('GT:')
            print(labels_list)
            print('Predictes:')
            print(predicted_list)
            return labels_list, predicted_list
            
        

    def print_confusion_matrix(y_true, y_pred, report=True):
        labels = sorted(list(set(y_true)))
        cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
        
        df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
     
        fig, ax = plt.subplots(figsize=(13, 11))
        plt.title("Macierz błędów", fontsize = 25)
        
        sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
        ax.set_ylim(len(set(y_true)), 0)
        plt.xlabel("Rzeczywiste klasy", fontsize = 20, labelpad=10)
        plt.ylabel("Predykowane klasy", fontsize = 20, labelpad=10)
        plt.show()
        
        if report:
            print('Classification Report')
            print(classification_report(y_true, y_pred))
            
            
    def learn_k_fold(kfolf_path,  save_path):
        # WEJSCIA: CIEZKA DO KATALOGU Z DANYMI OBRAZAMI DLA DANEGO FOLDA, SCIEZKA DO ZAPISU PLIKU
        
        ######################################################################
        # DEFINIUJ MODEL
        
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model = model.eval()
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
        model = model.to(device)
    
        #######################################################################
        # WCZYTANIE DANYCH TYLKO DO DANEGO FOLDA:
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(kfolf_path, x),
                                                  data_transforms[x])
                          for x in ['train', 'val' ]}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        
        # PARAMETRY NAUKI
        criterion = nn.CrossEntropyLoss()     
        optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.0001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        epochs = 100
        
        # UCZENIE MODELU Z TYMI PARAMETRAMI
        model = train_model(dataloaders, dataset_sizes, model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=epochs)
        model.eval()    
        
        if torch.cuda.is_available():
            model.cuda()
    
        # OCENA MODELU NA ZBIORZE TESTOWYM
        test_dataset = {x: datasets.ImageFolder(os.path.join(kfolf_path, x),
                                                  data_transforms[x])
                          for x in ['test']}
        
        test_dataloaders = {x: torch.utils.data.DataLoader(test_dataset[x], batch_size=1,
                                                     shuffle=True, num_workers=1)
                      for x in ['test']}
        
        labels_tensor, outputs_tensor = evaluate_trained_model(  test_dataloaders, model)
        labels_list, outputs_list = tensors_to_list(labels_tensor, outputs_tensor)
        print_confusion_matrix(labels_list, outputs_list)
        
        acc = metrics.accuracy_score(labels_list, outputs_list)
        
        torch.save(model.state_dict(), save_path)
        
        return acc
            
    # %% DATA TRANSFORMS
    input_shape = 254
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale =  260
    
   
    data_transforms = {
       'train': transforms.Compose([
           transforms.Resize(scale),
           transforms.RandomResizedCrop(input_shape),
           transforms.RandomHorizontalFlip(),
           transforms.RandomRotation(degrees=(-15,15)),
           transforms.ToTensor()
           # transforms.Normalize(mean, std)
       ]),
       'val': transforms.Compose([
           transforms.Resize(input_shape),
           # transforms.CenterCrop(input_shape),
           # transforms.RandomHorizontalFlip(),
           # transforms.RandomRotation(degrees=(-15,15)),
           transforms.ToTensor()
           # transforms.Normalize(mean, std)
       ]),
       'test': transforms.Compose([
           transforms.Resize(input_shape),
           # transforms.CenterCrop(input_shape),
           transforms.ToTensor()
           # transforms.Normalize(mean, std)
       ]),
    }
    
    #%%
    
    
#     acc1 =learn_k_fold(path_k1,  path_save_models+ '/model_alexnet_TF_k1.pt')
#     acc2 =learn_k_fold(path_k2,  path_save_models+'/model_alexnet_TF_k2.pt')        
#     acc3 =learn_k_fold(path_k3,  path_save_models+'/model_alexnet_TF_k3.pt')        
#     acc4 =learn_k_fold(path_k4,  path_save_models+'/model_alexnet_TF_k4.pt') 
#     acc5 =learn_k_fold(path_k5,  path_save_models+'/model_alexnet_TF_k5.pt')
    
# #%%
#     acc1 = acc1*100
#     acc2 = acc2*100
#     acc3 = acc3*100
#     acc4 = acc4*100
#     acc5 = acc5*100
#     print(f'Acc k1: {acc1:4f}')
#     print(f'Acc k2: {acc2:4f}')
#     print(f'Acc k3: {acc3:4f}')
#     print(f'Acc k4: {acc4:4f}')
#     print(f'Acc k5: {acc5:4f}')
#     avg_acc = np.mean([acc1, acc2, acc3, acc4, acc5])
#     print(f'Acc avg: {avg_acc:4f}')
    
    

# % WCZYTANIE DANYCH DO UCZENIA BEZ K-FOLDA
    data_dir = 'D:/HFUS'
    #model_path = 'D:/HFUS/modele_uczone_TF/model_resnet101_TF2.pt' 
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(path_k5, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test' ]}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                  shuffle=True, num_workers=8)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    #%%
    
    inputs, classes = next(iter(dataloaders['train']))
    
    imshow(torchvision.utils.make_grid(inputs))
  # %% TRANSFER LEARNING RESNET101:
    # model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, len(class_names))
    # model = model.to(device)

    
    #%%TRANSFER LEARNING DENSENET201
    # model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.classifier.in_features 
    # model.classifier = nn.Linear(num_ftrs, len(class_names))
    # model = model.to(device)
    
    #%%TRANSFER LEARNING SQUEEZENET 1_1
    # model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # model.classifier._modules["1"] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1))
    # model = model.to(device)
    
    
    #%%TRANSFER LEARNING ALEXNET
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    model = model.eval()
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    # %% TRANSFER LEARNING GOOGLENET:
    # model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, len(class_names))
    # model = model.to(device)
    
    # %% TRANSFER LEARNING INCEPTION:
    # model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, len(class_names))
    # model = model.to(device)
    
    # %% TRANSFER LEARNING VGG:
    # model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.classifier[-1].in_features 
    # model.classifier[-1] = nn.Linear(num_ftrs, len(class_names))
    # model = model.to(device)
# %% NAUKA
    criterion = nn.CrossEntropyLoss()     
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
      # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    epochs = 100
    model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler,
                            num_epochs=epochs)
    model.eval()    
    #%%
    if torch.cuda.is_available():
        model.cuda()

# %% TESTOWY WCZYTANIE
    inputs, labels = next(iter(dataloaders['test']))
    imshow(torchvision.utils.make_grid(inputs))
    #inputs, labels = inputs.to(device), labels.to(device)
    test_inputs, test_labels = inputs.cuda(), labels.cuda()


    #%% OCENA MODELU 
    labels_tensor, outputs_tensor = evaluate_trained_model(dataloaders, model)
    labels_list, outputs_list = tensors_to_list(labels_tensor, outputs_tensor)
    print_confusion_matrix(labels_list, outputs_list)
    #%%
    acc = metrics.accuracy_score(labels_list, outputs_list)
    print(acc*100)
#%% ZAPIS MODELU
    # torch.save(model.state_dict(), path_save_models+ '/model_alexnet_TF_k1_acc_8730.pt')
    # torch.save(model.state_dict(), path_save_models+ '/model_alexnet_TF_k2_acc_8730.pt')
    # torch.save(model.state_dict(), path_save_models+ '/model_alexnet_TF_k3_acc_8889.pt')
    # torch.save(model.state_dict(), path_save_models+ '/model_alexnet_TF_k4_acc_88095.pt')
    torch.save(model.state_dict(), path_save_models+ '/model_alexnet_TF_k5_acc_8413.pt')



    #torch.save(model.state_dict(), 'D:/HFUS/modele_uczone_TF/model_resnet101_TF_acc94_batch8.pt')
    #torch.save(model.state_dict(), 'D:/HFUS/modele_uczone_TF/model_densenet201_TF8.pt')
    #torch.save(model.state_dict(), 'D:/HFUS/modele_uczone_TF/model_squeezenet1_1__TF4_acc84.pt')
    #torch.save(model.state_dict(), 'D:/HFUS/modele_uczone_TF/model_alexnet_TF5_acc82.pt') # step 6, gamma 0.2
    #torch.save(model.state_dict(), 'D:/HFUS/modele_uczone_TF/model_googlenet_TF5_acc97.pt') # step 7, gamma 0.4
    # torch.save(model.state_dict(), 'D:/HFUS/modele_uczone_TF/model_vgg19_TF2.pt')
   
