clear all,close all
%% paths
path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/densenet/HM_densenet_TF5_k1/" ;
% path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/googlenet/HM_googlenet_TF6_k1/" ;
% path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/mobilenet/HM_mobilenet_TF5_k1/" ;
%path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/resnet/HM_resnet_TF6_k2/" ;

%method_list = ["AblationCAM/HM_255/", "EigenGradCAM/HM_255/", "GradCAM/HM_255/", "GradCAM++/HM_255/",  "HiResCAM/HM_255/", "ScoreCAM/HM_255/"];
% method_list = ["GradientShap/pos/", "IG/pos/", "Occlusion_w15/pos/","Occlusion_w30/pos/", "Occlusion_w45/pos/", "Saliency/pos/"]
%method_list = ["GradientShap/neg/", "IG/neg/", "Occlusion_w15/neg/","Occlusion_w30/neg/", "Occlusion_w45/neg/"];
method_list = ["GradientShap/abs/", "IG/abs/", "Occlusion_w15/abs/","Occlusion_w30/abs/", "Occlusion_w45/abs/"];
method = method_list(1)

path_to_hm_method_dir = strcat(path_to_hm, "AblationCAM/" , 'HMwithIMG_title/');
path_to_hm_method = strcat(path_to_hm , method );

path_to_slm = "D:/SLM/SLM_nowe/ALL/";
path_to_save_max = "D:/SLM/SLM_nowe/saveMax";

%%
files = dir(path_to_hm_method_dir);
files = files(3:end) ;

%%
HITRATE = struct('img_name', 0, 'hit_rate1', 0, 'hit_rate2', 0);
%HITRATE2 = struct('img_name', 0, 'hit_rate', 0);

for k=1:length(files)
    try
   m = imread(strcat(path_to_hm_method, files(k).name));

%%
% imshow(m)
% %%
% maxM = m==255;
% imshow(maxM)
%%
[rows,cols,vals] = find(m==255);

%% znajdx obraz slm dla maski
slm = imread(strcat(path_to_slm, files(k).name));

%%
% figure,
% subplot(1,2,1)
% imshow(slm)
% subplot(1,2,2)
% imshow(m)
%%
hit_rate1 = 0;
hit_rate2 = 0;

for i = 1:length(rows)
    hit_points_val(i) = slm(rows(i), cols(i));

    if  hit_points_val(i) == 255
        hit_rate1 = hit_rate1 + 3;
        hit_rate2 = hit_rate2 + 1;
    elseif hit_points_val(i) == 128
        hit_rate1 = hit_rate1 + 2;
    elseif hit_points_val(i) == 64
        hit_rate1 = hit_rate1 + 1;
    elseif hit_points_val(i) == 32
        hit_rate1 = hit_rate1 + 0;
    else
        hit_rate1 = hit_rate1 - 1;
        hit_rate2 = hit_rate2 - 1;
    end

end

hit_rate1 = hit_rate1 / length(rows);
hit_rate2 = hit_rate2 / length(rows);

HITRATE(k).img_name = files(k).name;
HITRATE(k).hit_rate1 = hit_rate1;
HITRATE(k).hit_rate2 = hit_rate2;

    catch 
        disp(files(k).name)
    end
end
%%
% maxMdb = uint8(maxM);
% %%
% maxonslm = maxMdb .* slm;
% imshow(maxonslm)
% hit_point = max(max(maxonslm))