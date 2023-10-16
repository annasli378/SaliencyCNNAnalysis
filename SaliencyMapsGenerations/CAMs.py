# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:15:44 2023

@author: annas
"""
# IMPORTS
import os
import torch
import cv2

from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import time
import numpy as np
import PIL
from PIL import Image

import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ustawienia
if __name__ == '__main__':
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #%% FUNKCJE
    def get_heatmap(img, heatmap):
      # wej:
      # img - obraz po wstępnym przetworzeniu żeby ładna heatmapa wyszła
      # heatmap - orginalna heatmapa otrzymana z cama
      # wyj -  ładna heatmapa gotowa do zapisania i wywietlenia sobie
      heatmap = cv2.resize(heatmap,(img.shape[1], img.shape[0]))
      heatmap = (heatmap*255).astype("uint8")
      return heatmap
    
    def get_heatmap_with_img(img, heatmap, alpha = 0.4):
      # wej:
      # img - obraz po wstępnym przetworzeniu żeby ładna heatmapa wyszła
      # heatmap - orginalna heatmapa otrzymana z cama
      # alpha - współczynnik mówiący jak bardzo przezroczysta ma być nałozona na obrazek heatmapa
      # wyj -  obrazek z nałożoną heatmapą
      heatmap = get_heatmap(img, heatmap)
      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
      superimposed_img = heatmap * alpha + img
      superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
      superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
      imgwithheat = Image.fromarray(superimposed_img)
      return imgwithheat
    
    def preprocess_image_to_show(img):
        target_size=(224, 224)
        img = cv2.resize(img, target_size)
        img = np.asarray(img)
        img = img/255
        return np.float32(img)
    
    
    def imshow(img):
            plt.imshow(np.transpose(img, (0, 1, 2)))
            plt.show()
            
    
    def tensors_to_list(all_labels, all_predicted):
        labels_list = []
        predicted_list = []
        if type(all_labels[0]) == torch.Tensor:
            #print('Zamiana listy tensorów na listę intów:')
            for tens in all_labels:
                tens_len = len(tens)
                for i in range(0,tens_len):
                    labels_list.append(tens[i].item())
            for tens in all_predicted:
                tens_len = len(tens)
                for i in range(0,tens_len):
                    predicted_list.append(tens[i].item())
            return labels_list, predicted_list    

        
    # def evaluate_trained_model(model, test_data, target_labels):
    #     correct = 0
    #     total = len(target_labels)
    #     all_predicted = []        
    #     for i in range(0,total):
    #         test_img = test_data[i]
    #         label = target_labels[i]
    #         test_imput = (torch.from_numpy(np.transpose(test_img, (2, 0, 1)))).unsqueeze_(0).cuda()
    #         outputs = model(test_imput)
    #         _, predicted = torch.max(outputs.data, 1)
    #         print(outputs)
             
            
    #         correct += (predicted == label).sum().item()
        
    #     print(f'Accuracy of the network on the 126 test images: {100 * correct // total} %')
        
    def evaluate_trained_model(model):
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
            
            
    def evaluate_on_test_dataset(model):
        labels_tensor, outputs_tensor = evaluate_trained_model( model)
        labels_list, outputs_list = tensors_to_list(labels_tensor, outputs_tensor)
        print_confusion_matrix(labels_list, outputs_list)
        
        return labels_list, outputs_list 
            
    
    def normalize_heat_map(heatmap):
        # wej - heatmapa orginalna
        # wyj - heatmapa znormalizowana do 1
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        return np.uint8(255 * heatmap)
    
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.transpose(tensor.cpu().numpy(), (2,3,0,1))
        return PIL.Image.fromarray(tensor)
    
    def get_and_save_cams(cam_metod, dataloaders, layer_name, path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names):
        since = time.time()
        cnt=0
        # GRADCAM lub inny CAM
        warstwa = layer_name # nazwa w formie tekstu zakończona / np. "layer4/"
        for data in dataloaders['test']:
                print(cnt)
                
                
                image, label = data
                image = image.to(device)
                label = label.to(device)
                #output = model(image)
                #_, predicted = torch.max(outputs.data, 1)    
                
                input_tensor = image.cuda()
                label_name = int(label[0].cpu().numpy())
                target = [ClassifierOutputTarget(label_name)]
                
                grayscale_cam1 = cam_metod(input_tensor=input_tensor)[0, :]
                grayscale_cam2 = cam_metod(input_tensor=input_tensor, targets = target)[0, :]
                
                grayscale_cam3 = cam_metod(input_tensor=input_tensor, aug_smooth=True)[0, :]
                grayscale_cam4 = cam_metod(input_tensor=input_tensor, targets = target, aug_smooth=True)[0, :]
                
                grayscale_cam5 = cam_metod(input_tensor=input_tensor, eigen_smooth=True)[0, :]
                grayscale_cam6 = cam_metod(input_tensor=input_tensor, targets = target, eigen_smooth=True)[0, :]

                grayscale_cam7 = cam_metod(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=True)[0, :]
                grayscale_cam8 = cam_metod(input_tensor=input_tensor,  targets = target, aug_smooth=True,eigen_smooth=True)[0, :]
                
                preprocessed_img = all_preprocessed_img[cnt]
                img_name = all_names[cnt]
                print(img_name)
                
                vis1 = show_cam_on_image(preprocessed_img, grayscale_cam1, use_rgb=False)
                vis2 = show_cam_on_image(preprocessed_img, grayscale_cam2, use_rgb=False)
                vis3 = show_cam_on_image(preprocessed_img, grayscale_cam3, use_rgb=False)
                vis4 = show_cam_on_image(preprocessed_img, grayscale_cam4, use_rgb=False)
                vis5 = show_cam_on_image(preprocessed_img, grayscale_cam5, use_rgb=False)
                vis6 = show_cam_on_image(preprocessed_img, grayscale_cam6, use_rgb=False)
                vis7 = show_cam_on_image(preprocessed_img, grayscale_cam7, use_rgb=False)
                vis8 = show_cam_on_image(preprocessed_img, grayscale_cam8, use_rgb=False)
                
                norm_gray_cam1 = normalize_heat_map(grayscale_cam1)
                norm_gray_cam2 = normalize_heat_map(grayscale_cam2)
                norm_gray_cam3 = normalize_heat_map(grayscale_cam3)
                norm_gray_cam4 = normalize_heat_map(grayscale_cam4)
                norm_gray_cam5 = normalize_heat_map(grayscale_cam5)
                norm_gray_cam6 = normalize_heat_map(grayscale_cam6)
                norm_gray_cam7 = normalize_heat_map(grayscale_cam7)
                norm_gray_cam8 = normalize_heat_map(grayscale_cam8)
                
                
                #sciezki do zapisow
                cv2.imwrite(path_to_save_target_obliczony+'org/HM/'+warstwa+img_name, grayscale_cam1)
                cv2.imwrite(path_to_save_target_podany+'org/HM/'+warstwa+img_name, grayscale_cam2)
                cv2.imwrite(path_to_save_target_obliczony+"aug_smooth/HM/"+warstwa +img_name, grayscale_cam3)
                cv2.imwrite(path_to_save_target_podany+"aug_smooth/HM/"+warstwa+img_name, grayscale_cam4)
                cv2.imwrite(path_to_save_target_obliczony+"eigen_smooth/HM/"+warstwa+img_name, grayscale_cam5)
                cv2.imwrite(path_to_save_target_podany+"eigen_smooth/HM/"+warstwa+img_name, grayscale_cam6)
                cv2.imwrite(path_to_save_target_obliczony+"smooth/HM/"+warstwa+img_name, grayscale_cam7)
                cv2.imwrite(path_to_save_target_podany+"smooth/HM/"+warstwa+img_name, grayscale_cam8)
                
                cv2.imwrite(path_to_save_target_obliczony+'org/HMwithIMG/'+warstwa+img_name, vis1)
                cv2.imwrite(path_to_save_target_podany+'org/HMwithIMG/'+warstwa+img_name, vis2)
                cv2.imwrite(path_to_save_target_obliczony+"aug_smooth/HMwithIMG/"+warstwa+img_name, vis3)
                cv2.imwrite(path_to_save_target_podany+"aug_smooth/HMwithIMG/"+warstwa+img_name, vis4)
                cv2.imwrite(path_to_save_target_obliczony+"eigen_smooth/HMwithIMG/"+warstwa+img_name, vis5)
                cv2.imwrite(path_to_save_target_podany+"eigen_smooth/HMwithIMG/"+warstwa+img_name, vis6)
                cv2.imwrite(path_to_save_target_obliczony+"smooth/HMwithIMG/"+warstwa+img_name, vis7)
                cv2.imwrite(path_to_save_target_podany+"smooth/HMwithIMG/"+warstwa+img_name, vis8)                
                
                cv2.imwrite(path_to_save_target_obliczony+'org/HM_255/'+warstwa+img_name, norm_gray_cam1)
                cv2.imwrite(path_to_save_target_podany+'org/HM_255/'+warstwa+img_name, norm_gray_cam2)
                cv2.imwrite(path_to_save_target_obliczony+"aug_smooth/HM_255/"+warstwa+img_name, norm_gray_cam3)
                cv2.imwrite(path_to_save_target_podany+"aug_smooth/HM_255/"+warstwa+img_name, norm_gray_cam4)
                cv2.imwrite(path_to_save_target_obliczony+"eigen_smooth/HM_255/"+warstwa+img_name, norm_gray_cam5)
                cv2.imwrite(path_to_save_target_podany+"eigen_smooth/HM_255/"+warstwa+img_name, norm_gray_cam6)
                cv2.imwrite(path_to_save_target_obliczony+"smooth/HM_255/"+warstwa+img_name, norm_gray_cam7)
                cv2.imwrite(path_to_save_target_podany+"smooth/HM_255/"+warstwa+img_name, norm_gray_cam8)                
                
                
                
                
                cnt = cnt+1
        time_elapsed = time.time() - since
        print(f'Maps saved in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #%% WCZYTANIE MODELU 
    # sciezka do konkretnego modelu - tutaj renet101
    model_path = 'D:/HFUS/modele_uczone_TF/model_resnet101_TF.pt'
    # specyfikacja jakiego typu to jest model
    model = models.resnet101() 
    # modyfikacja ostatniej warsty żeby architektura nam pasowała
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    # załadowanie zapisanych wag z pliku do naszej architektury 
    model.load_state_dict(torch.load(model_path))

    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
# %% DATA TRANSFORMS
    input_shape = 224
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    data_transforms = {
       'test': transforms.Compose([
           transforms.CenterCrop(input_shape),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
       ]),
    }
    
#% WCZYTANIE DANYCH ZE ZBIORU TESTOWEGO
    data_dir = 'D:/HFUS'
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [ 'test' ]}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                 shuffle=True, num_workers=1)
                  for x in [ 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in [ 'test']}
    class_names = image_datasets['test'].classes
    
#% TESTOWY WCZYTANIE i ocena
    inputs, labels = next(iter(dataloaders['test']))
    test_inputs, test_labels = inputs.cuda(), labels.cuda()
    labels_list, outputs_list = evaluate_on_test_dataset(model)
    
    # pętlą po całym zbiorze 
    for data in dataloaders['test']:
                images, labels = data
                #print(labels)
                images = images.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
#% WCZYTANIE OBRAZKÓW DO WYWIETLANIA               
    path_to_test_data = "D:/HFUS/test/"    
    class_names = os.listdir(path_to_test_data)
    all_test_images = []
    all_preprocessed_img = []
    all_targets = []
    all_names = []
    all_labels = []
    
    
    for c in class_names:
      path_to_class = path_to_test_data + c
      cc = 0
      if c == 'CG':
          cc = 1
      elif c == 'PS':
          cc = 2
      elif c == 'ST':
          cc = 3
      
      img_paths = os.listdir(path_to_class)
    
      for p in img_paths:
        img_path = path_to_class + "/"+p
        img = cv2.imread(img_path)
        img_proc = preprocess_image_to_show(img)
    
        all_preprocessed_img.append(img_proc)
        all_test_images.append(img)
        all_targets.append(c)
        all_names.append(p)
        all_labels.append(cc)
#%% USTAWIAMY CAMY!!!!!
# model 
# target_layers - można zmieniać !!! https://github.com/jacobgil/pytorch-grad-cam 

    grad_cam = GradCAM(
        model=model,
        target_layers=model.layer3,#[model.layer4[-1]],
        use_cuda=True
    )
    
    
    
#%% Wejciowe:
    # input_tensor -? torch.Torch
    # targets -> list
    # 0,1,2,3 ->????
    input_tensor = inputs.cuda()
    #input_tensor = (torch.from_numpy(np.transpose(all_preprocessed_img[0], (2, 0, 1)))).unsqueeze_(0).cuda()
    targets = [ClassifierOutputTarget(2)]#(torch.from_numpy(np.asarray(all_labels[50]))).cuda()
#%% GRADCAM
    preprocessed_img = all_preprocessed_img[-1]
    #image = tensor_to_image(input_tensor)
    
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets,eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(preprocessed_img, grayscale_cam, use_rgb=False)
    my_vis = get_heatmap_with_img(all_test_images[-1], grayscale_cam)
#%% pokaż:

    
    cv2.imshow('grayscale_cam',grayscale_cam)
    cv2.waitKey(0)

    cv2.imshow('visualization', visualization)
    cv2.waitKey(0)
#%%
    cv2.imwrite('D:/HFUS/testHM.png', grayscale_cam)
#%%
    warstwa = 'layer3/'
    path_to_save_target_podany = "D:/wyniki/resnet/GradCAM/target_podany/" 
    path_to_save_target_obliczony = "D:/wyniki/resnet/GradCAM/target_obliczony/"    
    
    get_and_save_cams(grad_cam, dataloaders, warstwa, path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
#%%
    grad_cam = GradCAM(
        model=model,
        target_layers=model.layer4,#[model.layer4[-1]],
        use_cuda=True
    )
    
    get_and_save_cams(grad_cam, dataloaders, "layer4/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
#%%    
    grad_cam = GradCAM(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )
    
    get_and_save_cams(grad_cam, dataloaders, "fc/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
#%%    
    grad_cam = GradCAM(
        model=model,
        target_layers=model.layer2,
        use_cuda=True
    )
    get_and_save_cams(grad_cam, dataloaders, "layer2/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
#%%    
    grad_cam = GradCAM(
        model=model,
        target_layers=model.layer1,
        use_cuda=True
    )
    get_and_save_cams(grad_cam, dataloaders, "layer1/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
#%%    
# =============================================================================
#     Dla całego zbioru danych generacja i zapis heatmap z gradcama
# =============================================================================
    # cnt = 0
    # warstwa = 'layer3/'
    # path_to_save_target_podany = "D:/wyniki/resnet/GradCAM/target_podany/" 
    # path_to_save_target_obliczony = "D:/wyniki/resnet/GradCAM/target_obliczony/" 
    
    # for data in dataloaders['test']:
    #             print(cnt)
                
                
    #             image, label = data
    #             image = image.to(device)
    #             label = label.to(device)
    #             output = model(image)
    #             _, predicted = torch.max(outputs.data, 1)    
                
    #             input_tensor = image.cuda()
    #             label_name = int(labels[0].cpu().numpy())
    #             target = [ClassifierOutputTarget(label_name)]
                
    #             grayscale_cam1 = grad_cam(input_tensor=input_tensor)[0, :]
    #             grayscale_cam2 = grad_cam(input_tensor=input_tensor, targets = target)[0, :]
                
    #             grayscale_cam3 = grad_cam(input_tensor=input_tensor, aug_smooth=True)[0, :]
    #             grayscale_cam4 = grad_cam(input_tensor=input_tensor, targets = target, aug_smooth=True)[0, :]
                
    #             grayscale_cam5 = grad_cam(input_tensor=input_tensor, eigen_smooth=True)[0, :]
    #             grayscale_cam6 = grad_cam(input_tensor=input_tensor, targets = target, eigen_smooth=True)[0, :]

    #             grayscale_cam7 = grad_cam(input_tensor=input_tensor, aug_smooth=True, eigen_smooth=True)[0, :]
    #             grayscale_cam8 = grad_cam(input_tensor=input_tensor,  targets = target, aug_smooth=True,eigen_smooth=True)[0, :]
                
    #             preprocessed_img = all_preprocessed_img[cnt]
    #             img_name = all_names[cnt]
    #             print(img_name)
                
    #             vis1 = show_cam_on_image(preprocessed_img, grayscale_cam1, use_rgb=False)
    #             vis2 = show_cam_on_image(preprocessed_img, grayscale_cam2, use_rgb=False)
    #             vis3 = show_cam_on_image(preprocessed_img, grayscale_cam3, use_rgb=False)
    #             vis4 = show_cam_on_image(preprocessed_img, grayscale_cam4, use_rgb=False)
    #             vis5 = show_cam_on_image(preprocessed_img, grayscale_cam5, use_rgb=False)
    #             vis6 = show_cam_on_image(preprocessed_img, grayscale_cam6, use_rgb=False)
    #             vis7 = show_cam_on_image(preprocessed_img, grayscale_cam7, use_rgb=False)
    #             vis8 = show_cam_on_image(preprocessed_img, grayscale_cam8, use_rgb=False)
                
    #             norm_gray_cam1 = normalize_heat_map(grayscale_cam1)
    #             norm_gray_cam2 = normalize_heat_map(grayscale_cam2)
    #             norm_gray_cam3 = normalize_heat_map(grayscale_cam3)
    #             norm_gray_cam4 = normalize_heat_map(grayscale_cam4)
    #             norm_gray_cam5 = normalize_heat_map(grayscale_cam5)
    #             norm_gray_cam6 = normalize_heat_map(grayscale_cam6)
    #             norm_gray_cam7 = normalize_heat_map(grayscale_cam7)
    #             norm_gray_cam8 = normalize_heat_map(grayscale_cam8)
                
                
    #             #sciezki do zapisow
    #             cv2.imwrite(path_to_save_target_obliczony+'org/HM/'+warstwa+img_name, grayscale_cam1)
    #             cv2.imwrite(path_to_save_target_podany+'org/HM/'+warstwa+img_name, grayscale_cam2)
    #             cv2.imwrite(path_to_save_target_obliczony+"aug_smooth/HM/"+warstwa +img_name, grayscale_cam3)
    #             cv2.imwrite(path_to_save_target_podany+"aug_smooth/HM/"+warstwa+img_name, grayscale_cam4)
    #             cv2.imwrite(path_to_save_target_obliczony+"eigen_smooth/HM/"+warstwa+img_name, grayscale_cam5)
    #             cv2.imwrite(path_to_save_target_podany+"eigen_smooth/HM/"+warstwa+img_name, grayscale_cam6)
    #             cv2.imwrite(path_to_save_target_obliczony+"smooth/HM/"+warstwa+img_name, grayscale_cam7)
    #             cv2.imwrite(path_to_save_target_podany+"smooth/HM/"+warstwa+img_name, grayscale_cam8)
                
    #             cv2.imwrite(path_to_save_target_obliczony+'org/HMwithIMG/'+warstwa+img_name, vis1)
    #             cv2.imwrite(path_to_save_target_podany+'org/HMwithIMG/'+warstwa+img_name, vis2)
    #             cv2.imwrite(path_to_save_target_obliczony+"aug_smooth/HMwithIMG/"+warstwa+img_name, vis3)
    #             cv2.imwrite(path_to_save_target_podany+"aug_smooth/HMwithIMG/"+warstwa+img_name, vis4)
    #             cv2.imwrite(path_to_save_target_obliczony+"eigen_smooth/HMwithIMG/"+warstwa+img_name, vis5)
    #             cv2.imwrite(path_to_save_target_podany+"eigen_smooth/HMwithIMG/"+warstwa+img_name, vis6)
    #             cv2.imwrite(path_to_save_target_obliczony+"smooth/HMwithIMG/"+warstwa+img_name, vis7)
    #             cv2.imwrite(path_to_save_target_podany+"smooth/HMwithIMG/"+warstwa+img_name, vis8)                
                
    #             cv2.imwrite(path_to_save_target_obliczony+'org/HM_255/'+warstwa+img_name, norm_gray_cam1)
    #             cv2.imwrite(path_to_save_target_podany+'org/HM_255/'+warstwa+img_name, norm_gray_cam2)
    #             cv2.imwrite(path_to_save_target_obliczony+"aug_smooth/HM_255/"+warstwa+img_name, norm_gray_cam3)
    #             cv2.imwrite(path_to_save_target_podany+"aug_smooth/HM_255/"+warstwa+img_name, norm_gray_cam4)
    #             cv2.imwrite(path_to_save_target_obliczony+"eigen_smooth/HM_255/"+warstwa+img_name, norm_gray_cam5)
    #             cv2.imwrite(path_to_save_target_podany+"eigen_smooth/HM_255/"+warstwa+img_name, norm_gray_cam6)
    #             cv2.imwrite(path_to_save_target_obliczony+"smooth/HM_255/"+warstwa+img_name, norm_gray_cam7)
    #             cv2.imwrite(path_to_save_target_podany+"smooth/HM_255/"+warstwa+img_name, norm_gray_cam8)                
                
                
                
                
    #             cnt = cnt+1
                
                
    
#%% EIGENCAM
    eigen_cam = EigenCAM(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )
#%%
    grayscale_eigencam = eigen_cam(input_tensor, eigen_smooth=True)[0, :, :]
    eigencam_image = show_cam_on_image(preprocessed_img, grayscale_eigencam, use_rgb=False)
    cv2.imshow('grayscale_eigencam',grayscale_eigencam)
    cv2.waitKey(0)

    cv2.imshow('eigencam_image', eigencam_image)
    cv2.waitKey(0)
    
    
#%%
    path_to_save_target_podany = "D:/wyniki/resnet/EigenCAM/target_podany/" 
    path_to_save_target_obliczony = "D:/wyniki/resnet/EigenCAM/target_obliczony/"   
#%%    
    eigen_cam = EigenCAM(
        model=model,
        target_layers=model.layer1,
        use_cuda=True
    )
    get_and_save_cams(eigen_cam, dataloaders, "layer1/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
#%%    
    eigen_cam = EigenCAM(
        model=model,
        target_layers=model.layer2,
        use_cuda=True
    )
    get_and_save_cams(eigen_cam, dataloaders, "layer2/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    eigen_cam = EigenCAM(
        model=model,
        target_layers=model.layer3,
        use_cuda=True
    )
    get_and_save_cams(eigen_cam, dataloaders, "layer3/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    eigen_cam = EigenCAM(
        model=model,
        target_layers=model.layer4,
        use_cuda=True
    )
    get_and_save_cams(eigen_cam, dataloaders, "layer4/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    eigen_cam = EigenCAM(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )
    get_and_save_cams(eigen_cam, dataloaders, "fc/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
            
    
    
#%% GRADCAM++
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )    
 
    grayscale_camplusplus = grad_cam_plusplus(input_tensor, eigen_smooth=True)[0, :, :]
    camplusplus_image = show_cam_on_image(preprocessed_img, grayscale_camplusplus, use_rgb=False)
    cv2.imshow('grayscale_camplusplus',grayscale_camplusplus)
    cv2.waitKey(0)

    cv2.imshow('camplusplus_image', camplusplus_image)
    cv2.waitKey(0)
    
    #%%
    path_to_save_target_podany = "D:/wyniki/resnet/GradCAM++/target_podany/" 
    path_to_save_target_obliczony = "D:/wyniki/resnet/GradCAM++/target_obliczony/"   
    
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=model.layer1,
        use_cuda=True
    )
    get_and_save_cams(grad_cam_plusplus, dataloaders, "layer1/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=model.layer2,
        use_cuda=True
    )
    get_and_save_cams(grad_cam_plusplus, dataloaders, "layer2/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=model.layer3,
        use_cuda=True
    )
    get_and_save_cams(grad_cam_plusplus, dataloaders, "layer3/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=model.layer1,
        use_cuda=True
    )
    get_and_save_cams(grad_cam_plusplus, dataloaders, "layer1/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=model.layer4,
        use_cuda=True
    )
    get_and_save_cams(grad_cam_plusplus, dataloaders, "layer4/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names) 
    
    grad_cam_plusplus = GradCAMPlusPlus(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )
    get_and_save_cams(grad_cam_plusplus, dataloaders, "fc/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
#%% HIRESCAM
    hirescam_cam = HiResCAM(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )    
 
    grayscale_hirescam_cam = hirescam_cam(input_tensor)[0, :, :]
    hirescam_cam_image = show_cam_on_image(all_preprocessed_img[0], grayscale_hirescam_cam, use_rgb=False)
    cv2.imshow('grayscale_scorecam',grayscale_hirescam_cam)
    cv2.waitKey(0)

    cv2.imshow('scorecam_image', hirescam_cam_image)
    cv2.waitKey(0)
    #%%
    path_to_save_target_podany = "D:/wyniki/resnet/HiResCAM/target_podany/" 
    path_to_save_target_obliczony = "D:/wyniki/resnet/HiResCAM/target_obliczony/"   
    
    score_cam = HiResCAM(
        model=model,
        target_layers=model.layer1,
        use_cuda=True
    )
    get_and_save_cams(score_cam, dataloaders, "layer1/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)

    score_cam = HiResCAM(
        model=model,
        target_layers=model.layer2,
        use_cuda=True
    )
    get_and_save_cams(score_cam, dataloaders, "layer2/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    score_cam = HiResCAM(
        model=model,
        target_layers=model.layer3,
        use_cuda=True
    )
    get_and_save_cams(score_cam, dataloaders, "layer3/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    score_cam = HiResCAM(
        model=model,
        target_layers=model.layer4,
        use_cuda=True
    )
    get_and_save_cams(score_cam, dataloaders, "layer4/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
    score_cam = HiResCAM(
        model=model,
        target_layers=[model.layer4[-1]],
        use_cuda=True
    )
    get_and_save_cams(score_cam, dataloaders, "fc/", path_to_save_target_obliczony, path_to_save_target_podany,all_preprocessed_img,all_names)
    
  