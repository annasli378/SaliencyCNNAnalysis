clear all,close all
%% paths
 path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/densenet/HM_densenet_TF5_k1/" ; model="densenet/";
% path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/googlenet/HM_googlenet_TF6_k1/" ;model="googlenet/";
% path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/mobilenet/HM_mobilenet_TF5_k1/" ;model="mobilenet/";
% path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/resnet/HM_resnet_TF6_k2/" ;model="resnet/";

%method_list = ["AblationCAM/", "EigenGradCAM/", "GradCAM/", "GradCAM++/", "GradientShap/", "HiResCAM/", "IG/","Occlusion_w15/","Occlusion_w30/", "Occlusion_w45/", "Saliency/","ScoreCAM/"]; 
method_list=["AblationCAM/","GradCAM++/","Occlusion_w45/"];
progi = ["t_95", "t_90","t_85","t_80","t_75","t_70","t_65","t_60","t_55","t_50"];
prog = progi(7);
jaccards = struct("m", 0, "prog", 0, "jac3", 0,"jac2", 0,"jac1", 0,"jac0", 0,"jac_1", 0) ;
jaccards_method = struct("method", 0 , "jaccards", 0);
%%
method = method_list(1)
path_to_slm = "D:/SLM/SLM_nowe/ALL/";
path_to_HMMask = strcat("D:/SLM/SLM_nowe/MASKI/", model,method, prog, "/");

files = dir(path_to_HMMask);
files = files(3:end) ;

jac3 = 0;jac2 = 0;jac1 = 0;jac0 = 0;jac_1 = 0;jac21 = 0;jac0_1 = 0;

% Jaccard
for k=1:length(files)
   try
   m = imread(strcat(path_to_HMMask, files(k).name));
   slm = imread(strcat(path_to_slm, files(k).name));
   slm = slm(:,:,1);

    slm3 = (slm==255);
    slm2 = (slm==128);
    slm1 = (slm==64);
    slm0 = (slm==32);
    slm_1 = (slm==0);

    jacc_slm3 = jaccard(slm3,m);
    jacc_slm2 = jaccard(slm2,m);
    jacc_slm1 = jaccard(slm1,m);
    jacc_slm0 = jaccard(slm0,m);
    jacc_slm_1 = jaccard(slm_1,m);


wyniki(k).name = files(k).name;
wyniki(k).abl = jacc_slm3;
    


 catch 
        disp(files(k).name)
    end
    
end

method = method_list(2)
path_to_slm = "D:/SLM/SLM_nowe/ALL/";
path_to_HMMask = strcat("D:/SLM/SLM_nowe/MASKI/", model,method, prog, "/");

files = dir(path_to_HMMask);
files = files(3:end) ;

jac3 = 0;jac2 = 0;jac1 = 0;jac0 = 0;jac_1 = 0;jac21 = 0;jac0_1 = 0;

% Jaccard
for k=1:length(files)
   try
   m = imread(strcat(path_to_HMMask, files(k).name));
   slm = imread(strcat(path_to_slm, files(k).name));
   slm = slm(:,:,1);

    slm3 = (slm==255);
    slm2 = (slm==128);
    slm1 = (slm==64);
    slm0 = (slm==32);
    slm_1 = (slm==0);

    jacc_slm3 = jaccard(slm3,m);
    jacc_slm2 = jaccard(slm2,m);
    jacc_slm1 = jaccard(slm1,m);
    jacc_slm0 = jaccard(slm0,m);
    jacc_slm_1 = jaccard(slm_1,m);


wyniki(k).name = files(k).name;
wyniki(k).gc = jacc_slm3;
    


 catch 
        disp(files(k).name)
    end
    
end

method = method_list(3)
path_to_slm = "D:/SLM/SLM_nowe/ALL/";
path_to_HMMask = strcat("D:/SLM/SLM_nowe/MASKI/", model,method, prog, "/");

files = dir(path_to_HMMask);
files = files(3:end) ;

jac3 = 0;jac2 = 0;jac1 = 0;jac0 = 0;jac_1 = 0;jac21 = 0;jac0_1 = 0;

% Jaccard
for k=1:length(files)
   try
   m = imread(strcat(path_to_HMMask, files(k).name));
   slm = imread(strcat(path_to_slm, files(k).name));
   slm = slm(:,:,1);

    slm3 = (slm==255);
    slm2 = (slm==128);
    slm1 = (slm==64);
    slm0 = (slm==32);
    slm_1 = (slm==0);

    jacc_slm3 = jaccard(slm3,m);
    jacc_slm2 = jaccard(slm2,m);
    jacc_slm1 = jaccard(slm1,m);
    jacc_slm0 = jaccard(slm0,m);
    jacc_slm_1 = jaccard(slm_1,m);


wyniki(k).name = files(k).name;
wyniki(k).occ = jacc_slm3;
    


 catch 
        disp(files(k).name)
    end
    
end
