# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:09:36 2023

@author: annas
"""

"""
Then you run the new modified "dark" image through the model, and check the new category scores.

The metrics are:

    (Smaller value is better) Drop in Confidence: What's the percentage drop of the condience ? (or 0 if the confidence increased).

The confidence is assumed to drop a bit since we're removing details.

    (Larger value is better) Increase in confidence: In how many of the cases did the confidence increase.

You might ask: why do we need two complementory metrics, why not just measure the average change in confidence. I'm not sure, I suspect that would be better.

This is a way of measuring the "fidelity" or "faithfulness" of the explanation. We want a good explanation to reflect the actual regions that the model is using.

https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/CAM%20Metrics%20And%20Tuning%20Tutorial.ipynb

"""

# IMPORTS
import os
import torch
import cv2
import PIL
from pytorch_grad_cam.utils.image import show_cam_on_image

import time
import numpy as np
from PIL import Image
from numpy import savetxt
import matplotlib.pyplot as plt


import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import GoogLeNet_Weights


import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

import torch.nn as nn

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ustawienia
if __name__ == '__main__':
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    #%% SCIEZKI    
   # DENSENET
    # path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/densenet/HM_densenet_TF5_k1/"
    # path_to_model = "D:/Testy_metod/nauczone_modele_kfold/densenet/model_densenet_TF5_k1.pt"  
    # path_to_dataset = "D:/dataset_for_kfold5/k1"
    # model_name ="densenet/"
    

    # # GOOGLENET
    # path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/googlenet/HM_googlenet_TF6_k1/"
    # path_to_model = "D:/Testy_metod/nauczone_modele_kfold/googlenet/model_googlenet_TF6_k1.pt"  
    # path_to_dataset = "D:/dataset_for_kfold5/k1"   
    # model_name ='googlenet/'
    
    
    # # MOBILENET
    # path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/mobilenet/HM_mobilenet_TF5_k1/"
    # path_to_model = "D:/Testy_metod/nauczone_modele_kfold/mobilenet/model_mobilenet_TF5_k1.pt"  
    # path_to_dataset = "D:/dataset_for_kfold5/k1"    
    # model_name ="mobilenet/" 
    
    # RESNET
    path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/resnet/HM_resnet_TF6_k2/"
    path_to_model = "D:/Testy_metod/nauczone_modele_kfold/resnet/model_resnet_TF6_k2.pt"  
    path_to_dataset = "D:/dataset_for_kfold5/k2"
    model_name =  "resnet/" #  #
    
    
    
    # path_to_df = path_to_HM_dir + method + 'wyniki_zacieniania_hm.csv'
    num_classes = 4
    
    
    path_to_HM_dir = "D:/Testy_metod/nauczone_modele_kfold/"+ model_name 

    
    method_cam_list = ["AblationCAM/", "EigenGradCAM/", "GradCAM/", "GradCAM++/", "HiResCAM/", "ScoreCAM/"]
    method_captum_list = ["GradientShap/", "IG/", "Occlusion_w15/", "Occlusion_w30/", "Occlusion_w45/", "Saliency/"]
    
    #%% FUNKCJE
    def get_files_list(path_to_files):
      all_files_paths = []
      all_img_names = []
    
      classes = os.listdir(path_to_files)
    
      for c in classes:
        class_path = path_to_files+ "/"+c
        class_files = os.listdir(class_path)
    
        for p in class_files:
          img_path = class_path + "/"+p
          all_img_names.append(p)
          all_files_paths.append(img_path)

      return all_files_paths, all_img_names
  
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
        target_size=(254, 254)
        img = cv2.resize(img, target_size)
        img = np.asarray(img)
        img = img/255
        return np.float32(img)
    
    
    def imshow(img):
            plt.imshow(np.transpose(img, (0, 1, 2)))
            plt.show()
            
    
    def tensors_to_list(all_labels):
        labels_list = []
        
        if type(all_labels[0]) == torch.Tensor:
            #print('Zamiana listy tensorów na listę intów:')
            for tens in all_labels:
                tens_len = len(tens)
                for i in range(0,tens_len):
                    labels_list.append(tens[i].item())
           
            return labels_list

        
        
    def evaluate_trained_model(model):
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        all_probs = []
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
                en, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predicted.append(predicted)
                all_labels.append(labels)
                all_probs.append(en)
        
        print(f'Accuracy of the network on the {len(all_labels)} test images: {100 * correct // total} %')
        return all_labels, all_predicted, all_probs
    
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
        #plt.savefig(path_to_save_HM + "confusion_matrix.png")
        plt.show()
        
        if report:
            print('Classification Report')
            print(classification_report(y_true, y_pred))
            
            
    def evaluate_on_test_dataset(model):
        labels_tensor, outputs_tensor, probabilites_tensor = evaluate_trained_model( model)
        #print(probabilites_tensor)
        
        
        labels_list = tensors_to_list(labels_tensor)
        outputs_list = tensors_to_list(outputs_tensor)
        probs_list= tensors_to_list( probabilites_tensor)
        print_confusion_matrix(labels_list, outputs_list)
        #savetxt(path_to_save_HM + 'predicted.csv', outputs_list, delimiter=',')
        #savetxt(path_to_save_HM + 'GT.csv', labels_list, delimiter=',')
        return labels_list, outputs_list , probs_list
    
 
          
            
    
    def normalize_heat_map(heatmap):
        # wej - heatmapa orginalna
        # wyj - heatmapa znormalizowana do 1
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        return np.uint8(255 * heatmap)
    
    def tensor_to_image(tens):
        tensornp = tens.cpu().numpy()
        tensornp = tensornp.squeeze()
        npimg = np.transpose(tensornp, (1,2,0))
        # npimg = npimg * 255
        
        # tensornp = tensor*255
        # tensor = np.transpose(tensor.cpu().numpy(), (2,3,0,1))
        return npimg #PIL.Image.fromarray(tensor)
    
    
  #%% WCZYTANIE MODELU 
    # # specyfikacja jakiego typu to jest model
    # model = models.densenet201() 
    # model.eval()
    # # modyfikacja ostatniej warsty żeby architektura nam pasowała
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, 4)
    # # załadowanie zapisanych wag z pliku do naszej architektury 
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    
    model = models.resnet101()
    model = model.eval()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
        
    
    # model = models.mobilenet_v2()
    # model.eval()
    # num_ftrs = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_ftrs, 4)
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
        
        
    # model = torchvision.models.alexnet(weights=None).to(device)
    # model = model.eval()
    # num_ftrs = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_ftrs, num_classes)  
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    
    
    # model = models.squeezenet1_1()
    # model = model.eval()
    # model.classifier._modules["1"] = nn.Conv2d(512, 4, kernel_size=(1, 1))
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    
    
    # model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    

    if torch.cuda.is_available():
        model.cuda()
        model.eval() 
# %% DATA TRANSFORMS
    input_shape =  (224, 224)
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    data_transforms = {
       'test': transforms.Compose([
           transforms.Resize(input_shape),
           transforms.ToTensor()
           # transforms.Normalize(mean, std)
       ]),
    }
    
#% WCZYTANIE DANYCH ZE ZBIORU TESTOWEGO
    image_datasets = {x: datasets.ImageFolder(os.path.join(path_to_dataset, x),
                                              data_transforms[x])
                      for x in [ 'test' ]}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                 shuffle=False, num_workers=1)
                  for x in [ 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in [ 'test']}
    class_names = image_datasets['test'].classes
    
    
    all_files_paths, all_img_names = get_files_list(path_to_dataset+"/test")
    
    # for param in model.parameters():
    #     param.requires_grad = True
        
    #%% EWALUACJA
    #   pewnoci dla kazdego obrazu oryginalnego
    labels_list, outputs_list, probs_list  = evaluate_on_test_dataset(model)
    # print(labels_list)
    # print(outputs_list)
    # print(probs_list)
    
    #%% Ocena dla działanie obrazkow po wymnozeniu img z hm i maską 
    #   zacienianie najważniejszych elementów i obserwowanie spadków pewnoci 
    # jako różnicy między pewnocią starą a nową, 0 jeli pewnoć nie spadła lub wzrosła
    
    for method in method_cam_list:
        print(method)
        path_to_HM = path_to_save_HM + method + 'HM_255/' # 'pos/'
        path_to_df = path_to_HM_dir + method[0:-1]  + '_wyniki_zacieniania_hm.csv'
    
    
        cnt = 0
        
    
        klasa_oryginalna = []
        en_oryginalna = []
    
        roznice_en_po_zacienieniu = []
        en_po_zacienieniu = []
        
        roznice_en_po_rozjasnieniu = []
        en_po_rozjasnieniu = []    
        
        
        with torch.no_grad():
                for data in dataloaders['test']:
                    images, labels = data
                    # print(all_img_names[cnt])
                    
                    hm = cv2.imread(path_to_HM + all_img_names[cnt])
                    try:
                        hm_tens = torch.from_numpy(np.transpose((hm/255), ( 2, 1, 0))).float()
                        
                        hm_tens = hm_tens.unsqueeze(0)
                        
                        hm_inv = 255 - hm
                        hm_inv_tens = torch.from_numpy(np.transpose((hm_inv/255), ( 2, 1, 0))).float()
                        hm_inv_tens = hm_inv_tens.unsqueeze(0)
                        
                        # zacienianie obszarów nieistotnych na obrazie
                        image_zacienione_obszary_nieistotne = torch.multiply(images, hm_tens)
                        
                        
                        # zacienianie obszarów najwazniejszych
                        image_zacienione_obszary_najwazniejsze = torch.multiply(images, hm_inv_tens)
                        
                        # plt.imshow(tensor_to_image(image_zacienione_obszary_nieistotne))
                        # plt.title("Obszar najważniejszy dla tej predykcji")
                        # plt.show()
        
                        # plt.imshow(tensor_to_image(image_zacienione_obszary_najwazniejsze))
                        # plt.title("Obszar mało ważny dla tej predykcji")
                        # plt.show()                
                        
                        
                        # Ocena oryginalnych
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        en, predicted = torch.max(outputs.data, 1)
                        pred_label = predicted.item()
                        # #print(predicted)
                        
                        #klasa_oryginalna.append(predicted)
                        en_oryginalna.append(en.item())
        
        
                          # Ocena po zacienieniu mapa
                        image_zacienione_obszary_nieistotne = image_zacienione_obszary_nieistotne.to(device)
                        outputs_zn = model(image_zacienione_obszary_nieistotne)
                        en_zn = outputs_zn[0][pred_label].item()
                        # #print(predicted)
                        en_po_zacienieniu.append(en_zn)              
                        
                        
                        # Ocena po rozjasnieniu mapa
                        image_zacienione_obszary_najwazniejsze = image_zacienione_obszary_najwazniejsze.to(device)
                        outputs_zw = model(image_zacienione_obszary_najwazniejsze)
                        en_zw = outputs_zw[0][pred_label].item()
                        # #print(predicted)
                        en_po_rozjasnieniu.append(en_zw)   
                        
                        roznica_zaciem = en.item() - en_zn
                        roznica_rozjas = en.item() - en_zw
                        
                        roznice_en_po_zacienieniu.append(roznica_zaciem)
                        roznice_en_po_rozjasnieniu.append(roznica_rozjas)
                    
                    except:
                        print(all_img_names[cnt])
                        en_po_zacienieniu.append([]) 
                        en_po_rozjasnieniu.append([])  
                        roznice_en_po_zacienieniu.append([])
                        roznice_en_po_rozjasnieniu.append([])
                    
                    cnt = cnt+1
                    
        #klasa_oryginalna = tensors_to_list(klasa_oryginalna)
        
        
        # zapis do csv
        df = pd.DataFrame(list(zip(*[en_oryginalna, en_po_zacienieniu, en_po_rozjasnieniu, roznice_en_po_zacienieniu, roznice_en_po_rozjasnieniu]))).add_prefix('Col')
        df.to_csv(path_to_df, index=False)
        print(df)
    
    #%%
    for method in method_captum_list:
        print(method)
        path_to_HM = path_to_save_HM + method +  'pos/'
        path_to_df = path_to_HM_dir + method[0:-1] + '_wyniki_zacieniania_hm.csv'
    
    
        cnt = 0
        
        klasa_oryginalna = []
        en_oryginalna = []
    
        roznice_en_po_zacienieniu = []
        en_po_zacienieniu = []
        
        roznice_en_po_rozjasnieniu = []
        en_po_rozjasnieniu = []    
            
        
        
        with torch.no_grad():
                for data in dataloaders['test']:
                    images, labels = data
                    # print(all_img_names[cnt])
                    
                    hm = cv2.imread(path_to_HM + all_img_names[cnt])
                    try:
                        hm_tens = torch.from_numpy(np.transpose((hm/255), ( 2, 1, 0))).float()
                        
                        hm_tens = hm_tens.unsqueeze(0)
                        
                        hm_inv = 255 - hm
                        hm_inv_tens = torch.from_numpy(np.transpose((hm_inv/255), ( 2, 1, 0))).float()
                        hm_inv_tens = hm_inv_tens.unsqueeze(0)
                        
                        # zacienianie obszarów nieistotnych na obrazie
                        image_zacienione_obszary_nieistotne = torch.multiply(images, hm_tens)
                        
                        
                        # zacienianie obszarów najwazniejszych
                        image_zacienione_obszary_najwazniejsze = torch.multiply(images, hm_inv_tens)
                        
                        # plt.imshow(tensor_to_image(image_zacienione_obszary_nieistotne))
                        # plt.title("Obszar najważniejszy dla tej predykcji")
                        # plt.show()
        
                        # plt.imshow(tensor_to_image(image_zacienione_obszary_najwazniejsze))
                        # plt.title("Obszar mało ważny dla tej predykcji")
                        # plt.show()                
                        
                        
                        # Ocena oryginalnych
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        en, predicted = torch.max(outputs.data, 1)
                        pred_label = predicted.item()
                        # #print(predicted)
                        
                        #klasa_oryginalna.append(predicted)
                        en_oryginalna.append(en.item())
        
        
                          # Ocena po zacienieniu mapa
                        image_zacienione_obszary_nieistotne = image_zacienione_obszary_nieistotne.to(device)
                        outputs_zn = model(image_zacienione_obszary_nieistotne)
                        en_zn = outputs_zn[0][pred_label].item()
                        # #print(predicted)
                        en_po_zacienieniu.append(en_zn)              
                        
                        
                        # Ocena po rozjasnieniu mapa
                        image_zacienione_obszary_najwazniejsze = image_zacienione_obszary_najwazniejsze.to(device)
                        outputs_zw = model(image_zacienione_obszary_najwazniejsze)
                        en_zw = outputs_zw[0][pred_label].item()
                        # #print(predicted)
                        en_po_rozjasnieniu.append(en_zw)   
                        
                        roznica_zaciem = en.item() - en_zn
                        roznica_rozjas = en.item() - en_zw
                        
                        roznice_en_po_zacienieniu.append(roznica_zaciem)
                        roznice_en_po_rozjasnieniu.append(roznica_rozjas)
                    
                    except:
                        print(all_img_names[cnt])
                        en_po_zacienieniu.append([]) 
                        en_po_rozjasnieniu.append([])  
                        roznice_en_po_zacienieniu.append([])
                        roznice_en_po_rozjasnieniu.append([])
                    
                    cnt = cnt+1
                    
        #klasa_oryginalna = tensors_to_list(klasa_oryginalna)
        
        # zapis do csv
        df = pd.DataFrame(list(zip(*[en_oryginalna, en_po_zacienieniu, en_po_rozjasnieniu, roznice_en_po_zacienieniu, roznice_en_po_rozjasnieniu]))).add_prefix('Col')
        df.to_csv(path_to_df, index=False)
        print(df)
    
    
