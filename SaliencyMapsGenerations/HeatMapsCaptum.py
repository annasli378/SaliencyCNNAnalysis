# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:08:55 2023

@author: annas
"""

import numpy as np
import os

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchvision.models import GoogLeNet_Weights

import cv2

from captum.attr import visualization as viz
from captum.attr import (
    
    IntegratedGradients,
    Occlusion,
    Saliency,
    GradientShap,
    LRP
)


#%%
if __name__ == '__main__':
    
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
    def test_input_tensor_to_image(input_tensor):
        img = input_tensor / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        
        return np.transpose(npimg, (1, 2, 0))
    
    def normalize_heat_map(heatmap):
        # wej - heatmapa orginalna
        # wyj - heatmapa znormalizowana do 1
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap / np.max(heatmap)
        return np.uint8(255 * heatmap)
    
    def generate_HM(atrr, img, saving_path):
        # wszystkie atrybucje
        _, _, norm1, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
        
        
        # tylko ujemne
        _, _, norm2, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="negative", show_colorbar=True, title="Overlayed Integrated Gradients")

        
        # dodatnie
        _, _, norm3, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="positive", show_colorbar=True, title="Overlayed Integrated Gradients")
        
        # abs
        _, _, norm4, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title="Overlayed Integrated Gradients")

        # normalizacja 0-255
        
        
        
    #%% SCIEZKI
    
    # DENSENET
    path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/densenet/HM_densenet_TF5_k1/"
    path_to_model = "D:/Testy_metod/nauczone_modele_kfold/densenet/model_densenet_TF5_k1.pt"  
    path_to_dataset = "D:/dataset_for_kfold5/k1"

    

    # # GOOGLENET
    # path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/googlenet/HM_googlenet_TF6_k1/"
    # path_to_model = "D:/Testy_metod/nauczone_modele_kfold/googlenet/model_googlenet_TF6_k1.pt"  
    # path_to_dataset = "D:/dataset_for_kfold5/k1"   
    
    
    # # MOBILENET
    # path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/mobilenet/HM_mobilenet_TF5_k1/"
    # path_to_model = "D:/Testy_metod/nauczone_modele_kfold/mobilenet/model_mobilenet_TF5_k1.pt"  
    # path_to_dataset = "D:/dataset_for_kfold5/k1"    
    
    # # RESNET
    # path_to_save_HM ="D:/Testy_metod/nauczone_modele_kfold/resnet/HM_resnet_TF6_k2/"
    # path_to_model = "D:/Testy_metod/nauczone_modele_kfold/resnet/model_resnet_TF6_k2.pt"  
    # path_to_dataset = "D:/dataset_for_kfold5/k2"

    num_classes = 4
    
    
    
    
    model = models.densenet201() 
    # modyfikacja ostatniej warsty żeby architektura nam pasowała
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 4)
    # załadowanie zapisanych wag z pliku do naszej architektury 
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    
    # model = models.resnet101()
    # model = model.eval()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)    
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    
    
        
    # model = models.mobilenet_v2()
    # model.eval()
    # num_ftrs = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_ftrs, 4)
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    
    
    # model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    # model = model.eval()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()
    
    
    
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
    

    model.eval()
    
    #%% INTEGRATED GRADIENTS
    # torch.manual_seed(123)
    # np.random.seed(123)
    
    # input = torch.rand(2, 3)
    # baseline = torch.zeros(2, 3)
    
    # ig = IntegratedGradients(model)
    # attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
    # print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)
    
    cnt = 0
    
    print('Integrated Gradients')
    baseline =  torch.zeros([1, 3, 224, 224]) #.cuda()
    ig = IntegratedGradients(model)
    
    for data in dataloaders['test']:
            image, label = data
            print(cnt)
            
            # image = image.to(device)
            # label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)    
            predicted_label = predicted.numpy()
                # print(predicted[0])
            input_tensor = image #.cuda()
            attr_ig, delta = ig.attribute(input_tensor, baseline, target=predicted, return_convergence_delta=True,)
            
            original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
            attr_ig = np.transpose(attr_ig.squeeze().numpy(), (1, 2, 0))
            
            tit = 'IG: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
            pth = path_to_save_HM+'IG/neg/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'IG/negIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'IG/pos/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'IG/posIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'IG/abs/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'IG/absIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
            
            
            
            
            cnt = cnt+1
           
            
           
            
           
            
           
#     #%% OCCLISION
#     cnt = 0
    
#     print('Occlusion')
    
#     occlusion = Occlusion(model)
    
#     strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower
#     sliding_window_shapes=(3,45, 45)  # choose size enough to change object appearance
#     baseline = 0                     # values to occlude the image with. 0 corresponds to gray


    
    
#     for data in dataloaders['test']:
#             image, label = data
#             print(cnt)

#             output = model(image)
#             _, predicted = torch.max(output.data, 1)    
#             predicted_label = predicted.numpy()
            
#             input_tensor = image #.cuda()
#             attr_occ = occlusion.attribute(input_tensor, strides = strides,target=predicted,sliding_window_shapes=sliding_window_shapes,baselines=baseline)
            
#             original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
#             attr_occ = np.transpose(attr_occ.squeeze().numpy(), (1, 2, 0))
            
#             tit = 'Occ: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
#             try:
#                 if attr_occ.min() < 0 :
#                     pth = path_to_save_HM+'Occlusion_w45/neg/'+all_img_names[cnt]
#                     pthIMG = path_to_save_HM+'Occlusion_w45/negIMG/'+all_img_names[cnt]
#                     viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
#                     norm = normalize_heat_map(norm)
#                     cv2.imwrite(pth, norm)
#                     viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
                
#                 pth = path_to_save_HM+'Occlusion_w45/pos/'+all_img_names[cnt]
#                 pthIMG = path_to_save_HM+'Occlusion_w45/posIMG/'+all_img_names[cnt]
#                 viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
#                 norm = normalize_heat_map(norm)
#                 cv2.imwrite(pth, norm)
#                 viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
                
#                 pth = path_to_save_HM+'Occlusion_w45/abs/'+all_img_names[cnt]
#                 pthIMG = path_to_save_HM+'Occlusion_w45/absIMG/'+all_img_names[cnt]
#                 viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
#                 norm = normalize_heat_map(norm)
#                 cv2.imwrite(pth, norm)
#                 viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
#             except:
#                 print(all_img_names[cnt])
                
                        
            
            
            
            
#             cnt = cnt+1
            
#     #%% OCCLISION
#     cnt = 0
    
#     print('Occlusion')
    
#     occlusion = Occlusion(model)
    
#     strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower
#     sliding_window_shapes=(3,30, 30)  # choose size enough to change object appearance
#     baseline = 0                     # values to occlude the image with. 0 corresponds to gray


    
    
#     for data in dataloaders['test']:
#             image, label = data
#             print(cnt)

#             output = model(image)
#             _, predicted = torch.max(output.data, 1)    
#             predicted_label = predicted.numpy()
            
#             input_tensor = image #.cuda()
#             attr_occ = occlusion.attribute(input_tensor, strides = strides,target=predicted,sliding_window_shapes=sliding_window_shapes,baselines=baseline)
            
#             original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
#             attr_occ = np.transpose(attr_occ.squeeze().numpy(), (1, 2, 0))
            
#             tit = 'Occ: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
#             if attr_occ.min() < 0 :
#                 pth = path_to_save_HM+'Occlusion_w30/neg/'+all_img_names[cnt]
#                 pthIMG = path_to_save_HM+'Occlusion_w30/negIMG/'+all_img_names[cnt]
#                 viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
#                 norm = normalize_heat_map(norm)
#                 cv2.imwrite(pth, norm)
#                 viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
#             pth = path_to_save_HM+'Occlusion_w30/pos/'+all_img_names[cnt]
#             pthIMG = path_to_save_HM+'Occlusion_w30/posIMG/'+all_img_names[cnt]
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
#             norm = normalize_heat_map(norm)
#             cv2.imwrite(pth, norm)
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
#             pth = path_to_save_HM+'Occlusion_w30/abs/'+all_img_names[cnt]
#             pthIMG = path_to_save_HM+'Occlusion_w30/absIMG/'+all_img_names[cnt]
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
#             norm = normalize_heat_map(norm)
#             cv2.imwrite(pth, norm)
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
            
            
            
            
#             cnt = cnt+1
     
# #% OCCLISION
    cnt = 0
    
    print('Occlusion')
    
    occlusion = Occlusion(model)
    
    strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower
    sliding_window_shapes=(3,15, 15)  # choose size enough to change object appearance
    baseline = 0                     # values to occlude the image with. 0 corresponds to gray


    
    
    for data in dataloaders['test']:
            image, label = data
            print(cnt)

            output = model(image)
            _, predicted = torch.max(output.data, 1)    
            predicted_label = predicted.numpy()
            
            input_tensor = image #.cuda()
            attr_occ = occlusion.attribute(input_tensor, strides = strides,target=predicted,sliding_window_shapes=sliding_window_shapes,baselines=baseline)
            
            original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
            attr_occ = np.transpose(attr_occ.squeeze().numpy(), (1, 2, 0))
            
            tit = 'Occ: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
            if attr_occ.min() < 0 :
                pth = path_to_save_HM+'Occlusion_w15/neg/'+all_img_names[cnt]
                pthIMG = path_to_save_HM+'Occlusion_w15/negIMG/'+all_img_names[cnt]
                viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
                norm = normalize_heat_map(norm)
                cv2.imwrite(pth, norm)
                viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'Occlusion_w15/pos/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'Occlusion_w15/posIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'Occlusion_w15/abs/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'Occlusion_w15/absIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_occ, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
            
            
            
            
            cnt = cnt+1
    #% SALIENCY
    cnt = 0
    
    print('Saliency')
    
    saliency = Saliency(model)

    
    
    for data in dataloaders['test']:
            image, label = data
            print(cnt)

            output = model(image)
            _, predicted = torch.max(output.data, 1)    
            predicted_label = predicted.numpy()
            
            input_tensor = image #.cuda()
            attr_sal = saliency.attribute(input_tensor, target=predicted)
            
            original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
            attr_sal = np.transpose(attr_sal.squeeze().numpy(), (1, 2, 0))
            
            tit = 'Sal: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
            # pth = path_to_save_HM+'Saliency/neg/'+all_img_names[cnt]
            # pthIMG = path_to_save_HM+'Saliency/negIMG/'+all_img_names[cnt]
            # viz_ig, _, norm, _ = viz.visualize_image_attr(attr_sal, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
            # norm = normalize_heat_map(norm)
            # cv2.imwrite(pth, norm)
            # viz_ig, _, norm, _ = viz.visualize_image_attr(attr_sal, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'Saliency/pos/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'Saliency/posIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_sal, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_sal, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            # pth = path_to_save_HM+'Saliency/abs/'+all_img_names[cnt]
            # pthIMG = path_to_save_HM+'Saliency/absIMG/'+all_img_names[cnt]
            # viz_ig, _, norm, _ = viz.visualize_image_attr(attr_sal, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
            # norm = normalize_heat_map(norm)
            # cv2.imwrite(pth, norm)
            # viz_ig, _, norm, _ = viz.visualize_image_attr(attr_sal, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
            
            
            
            
            cnt = cnt+1
    
    
    
    #% GRADIENTSHAP
    cnt = 0
    
    print('GradientShap')
    
    gshap = GradientShap(model)
    baseline = torch.randn([1, 3, 224, 224]) * 0.001 # choosing baselines randomly

    
    for data in dataloaders['test']:
            image, label = data
            print(cnt)

            output = model(image)
            _, predicted = torch.max(output.data, 1)    
            predicted_label = predicted.numpy()
            input_tensor = image #.cuda()
            original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
            attr_gsh = gshap.attribute(input_tensor,target=predicted,baselines=baseline, stdevs=0.09, n_samples=5)                        
            attr_gsh = np.transpose(attr_gsh.squeeze().numpy(), (1, 2, 0))
            
            tit = 'GradShap: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
            pth = path_to_save_HM+'GradientShap/neg/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'GradientShap/negIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_gsh, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_gsh, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'GradientShap/pos/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'GradientShap/posIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_gsh, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_gsh, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
            pth = path_to_save_HM+'GradientShap/abs/'+all_img_names[cnt]
            pthIMG = path_to_save_HM+'GradientShap/absIMG/'+all_img_names[cnt]
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_gsh, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
            norm = normalize_heat_map(norm)
            cv2.imwrite(pth, norm)
            viz_ig, _, norm, _ = viz.visualize_image_attr(attr_gsh, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
            
            
            
            
            cnt = cnt+1
            
            
    #%% LRP
#     cnt = 0
    
#     print('LRP')
    
#     lrp = LRP(model)
   
# #%%
    
#     for data in dataloaders['test']:
#             image, label = data
#             print(cnt)

#             output = model(image)
#             _, predicted = torch.max(output.data, 1)    
#             predicted_label = predicted.numpy()
#             input_tensor = image #.cuda()
#             original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
#             attr_lrp  = lrp.attribute(input_tensor, target=3)  
            
#             attr_lrp = np.transpose(attr_lrp.squeeze().detach().numpy(), (1, 2, 0))
            
#             tit = 'LRP: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
#             pth = path_to_save_HM+'LRP/neg/'+all_img_names[cnt]
#             pthIMG = path_to_save_HM+'LRP/negIMG/'+all_img_names[cnt]
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_lrp, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
#             norm = normalize_heat_map(norm)
#             cv2.imwrite(pth, norm)
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_lrp, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
#             pth = path_to_save_HM+'LRP/pos/'+all_img_names[cnt]
#             pthIMG = path_to_save_HM+'LRP/posIMG/'+all_img_names[cnt]
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_lrp, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
#             norm = normalize_heat_map(norm)
#             cv2.imwrite(pth, norm)
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_lrp, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
#             pth = path_to_save_HM+'LRP/abs/'+all_img_names[cnt]
#             pthIMG = path_to_save_HM+'LRP/absIMG/'+all_img_names[cnt]
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_lrp, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
#             norm = normalize_heat_map(norm)
#             cv2.imwrite(pth, norm)
#             viz_ig, _, norm, _ = viz.visualize_image_attr(attr_lrp, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
            
            
#             break
            
#             cnt = cnt+1
            
            
    #%% SHAPLEY VALUE SAMPLING
    # cnt = 0
    
    # print('Shapley Value Sampling')
    
    # shap_vs = ShapleyValueSampling(model)
    # # baseline = torch.randn([1, 3, 224, 224]) * 0.001 # choosing baselines randomly

    
    # for data in dataloaders['test']:
    #         image, label = data
    #         print(cnt)

    #         output = model(image)
    #         _, predicted = torch.max(output.data, 1)    
    #         predicted_label = predicted.numpy()
    #         input_tensor = image #.cuda()
    #         original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
            
    #         attr_shap = shap_vs.attribute(input_tensor,target=predicted,perturbations_per_eval = 5, n_samples  = 5, show_progress  = True)                        
    #         attr_shap = np.transpose(attr_shap.squeeze().numpy(), (1, 2, 0))
            
    #         tit = 'ShapVal: Class predicted: '+ str(predicted_label[0]) + ', with score '+  "{:.2f}".format(float(output[0][predicted[0]]) )
            
    #         pth = path_to_save_HM+'ShapleyValueSampling/neg/'+all_img_names[cnt]
    #         pthIMG = path_to_save_HM+'ShapleyValueSampling/negIMG/'+all_img_names[cnt]
    #         viz_ig, _, norm, _ = viz.visualize_image_attr(attr_shap, original_image, method="heat_map",sign="negative", show_colorbar=True, title=tit)
    #         norm = normalize_heat_map(norm)
    #         cv2.imwrite(pth, norm)
    #         viz_ig, _, norm, _ = viz.visualize_image_attr(attr_shap, original_image, method="blended_heat_map",sign="negative", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
    #         pth = path_to_save_HM+'ShapleyValueSampling/pos/'+all_img_names[cnt]
    #         pthIMG = path_to_save_HM+'ShapleyValueSampling/posIMG/'+all_img_names[cnt]
    #         viz_ig, _, norm, _ = viz.visualize_image_attr(attr_shap, original_image, method="heat_map",sign="positive", show_colorbar=True, title=tit)
    #         norm = normalize_heat_map(norm)
    #         cv2.imwrite(pth, norm)
    #         viz_ig, _, norm, _ = viz.visualize_image_attr(attr_shap, original_image, method="blended_heat_map",sign="positive", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
    #         pth = path_to_save_HM+'ShapleyValueSampling/abs/'+all_img_names[cnt]
    #         pthIMG = path_to_save_HM+'ShapleyValueSampling/absIMG/'+all_img_names[cnt]
    #         viz_ig, _, norm, _ = viz.visualize_image_attr(attr_shap, original_image, method="heat_map",sign="absolute_value", show_colorbar=True, title=tit)
    #         norm = normalize_heat_map(norm)
    #         cv2.imwrite(pth, norm)
    #         viz_ig, _, norm, _ = viz.visualize_image_attr(attr_shap, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title=tit, path_to_save = pthIMG)
            
                        
    #         break
            
            
            
    #         cnt = cnt+1         
    #%% INTEGRATED GRADIENTS
    
    
    #input = image #torch.rand(2, 3)
    # baseline =  torch.zeros([1, 3, 254, 254])
    # 
    # ig = IntegratedGradients(model)
    
    # attributions, delta = ig.attribute(image, baseline, target=3, return_convergence_delta=True,)
    # print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)
    
    # #%%
    # original_image = np.transpose((image.squeeze().numpy() / 2) + 0.5, (1, 2, 0))
    
    # plt.imshow(original_image)
    # plt.show()
    # #%%
    # _ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")
    # #%%
    # attr_ig = np.transpose(attributions.squeeze().numpy(), (1, 2, 0))
    # viz_ig, _, norm, hm = viz.visualize_image_attr(attr_ig, original_image, method="heat_map",sign="negative", show_colorbar=True, title="Overlayed Integrated Gradients")
    # print(norm)
    
    
    # cv2.imshow('vis1', norm)
    # cv2.waitKey()
    
    
    
    # #%%
    # plt.imshow(normalize_heat_map(attr_ig))
    # plt.show()
    
    #%%
    # nt = NoiseTunnel(ig)
    # attr_ig_nt = nt.attribute(image, nt_type='smoothgrad',
    #                                         nt_samples=2, target=3)
    # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    # #%%
    # _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="heat_map", sign="all", 
    #                          outlier_perc=10, show_colorbar=True, 
    #                          title="Overlayed Integrated Gradients \n with SmoothGrad")
    #%% DEEPLIFT
    # dl = DeepLift(model)
    # attr_dl, delta = dl.attribute(image, baseline, target=3, return_convergence_delta=True)
    # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    # print('IG Attributions:', attr_dl)
    # print('Convergence Delta:', delta)
    
    # #%%
    # _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="absolute_value", show_colorbar=True, title="Overlayed DeepLIFT")
    
    #%% OCCLUSION
    
    # occlusion = Occlusion(model)
    
    # strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower
    # target=3,                       #  index 
    # sliding_window_shapes=(3,45, 45)  # choose size enough to change object appearance
    # baselines = 0                     # values to occlude the image with. 0 corresponds to gray

    # attribution_occ = occlusion.attribute(image,
    #                                    strides = strides,
    #                                    target=target,
    #                                    sliding_window_shapes=sliding_window_shapes,
    #                                    baselines=baselines)
    
    #%%
#     attr_oc = np.transpose(attribution_occ.squeeze().cpu().detach().numpy(), (1,2,0))
    
#     vis_types = ["heat_map", "original_image"]
#     vis_signs = ["all", "all"] # "positive", "negative", or "all" to show both
# # positive attribution indicates that the presence of the area increases the prediction score
# # negative attribution indicates distractor areas whose absence increases the score

#     viz_oc, _, norm, hm = viz.visualize_image_attr(attr_oc, original_image, method="blended_heat_map",sign="all", show_colorbar=True, title="Overlayed Occlusuon")
    
#     #%% LIME
#     from captum.attr._core.lime import get_exp_kernel_similarity_function
#     from captum.attr import Lime, LimeBase
#     from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

#     #%%
#     exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=50)

#     lr_lime = Lime(
#         model, 
#         interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
#         similarity_func=exp_eucl_distance
#     )
        
#     #%%
#     attr_lime = lr_lime.attribute(
#         image,
#         target=3,
#         n_samples=40,
#         perturbations_per_eval=4,
#         show_progress=True
#     ).squeeze(0)

#     print('Attribution range:', attr_lime.min().item(), 'to', attr_lime.max().item())
    
#     #%%
    
#     viz_lime,_,_,_ = viz.visualize_image_attr(
#         attr_lime.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels 
#         original_image,
#         method='blended_heat_map',
#         sign='all',
#         show_colorbar=True )
    
#     #%%
#     svs = ShapleyValueSampling(model)
    
#     attr_svs = svs.attribute(image, target=3, n_samples=200)
        
        
        
    