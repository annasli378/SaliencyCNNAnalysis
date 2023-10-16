clear all,close all
%% paths
%  path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/densenet/HM_densenet_TF5_k1/" ;
% path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/googlenet/HM_googlenet_TF6_k1/" ;
%path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/mobilenet/HM_mobilenet_TF5_k1/" ;
 path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/resnet/HM_resnet_TF6_k2/" ;

method_list = ["AblationCAM/", "EigenGradCAM/", "GradCAM/", "GradCAM++/", "GradientShap/", "HiResCAM/", "IG/","Occlusion_w15/","Occlusion_w30/", "Occlusion_w45/", "Saliency/","ScoreCAM/"]; 

progi = ["t_95", "t_90","t_85","t_80","t_75","t_70","t_65","t_60","t_55","t_50"];

jaccards = struct("m", 0, "prog", 0, "jac3", 0,"jac2", 0,"jac1", 0,"jac0", 0,"jac_1", 0) ;
jaccards_method = struct("method", 0 , "jaccards", 0);
%%
cnt = 1;
for met = 1:length(method_list)
    method = method_list(met)
    for t =1:length(progi)
        prog = progi(t);
        path_to_slm = "D:/SLM/SLM_nowe/ALL/";
        path_to_HMMask = strcat("D:/SLM/SLM_nowe/MASKI/resnet/", method, prog, "/");
        
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
%             slm21 = slm2&slm1;
%             slm0_1 = slm0 & slm_1;
        
            jacc_slm3 = jaccard(slm3,m);
            jacc_slm2 = jaccard(slm2,m);
            jacc_slm1 = jaccard(slm1,m);
            jacc_slm0 = jaccard(slm0,m);
            jacc_slm_1 = jaccard(slm_1,m);
%             jacc_slm21 = jaccard(slm21,m);
%             jacc_slm0_1 = jaccard(slm0_1,m);
    
            jac3 = jac3+jacc_slm3;
            jac2 = jac2+jacc_slm2;
            jac1 = jac1+jacc_slm1;
            jac0 = jac0+jacc_slm0;
            jac_1 = jac_1+jacc_slm_1;
%             jac21 = jac21+jacc_slm21;
%             jac0_1 = jac0_1+jacc_slm0_1;
    
    
            
    
    
         catch 
                disp(files(k).name)
            end
            
        end
    
        jac3 = jac3/length(files);
        jac2 = jac2/length(files);
        jac1 = jac1/length(files);
        jac0 = jac0/length(files);
        jac_1 = jac_1/length(files);
%         jac21 = jac21/length(files);
%         jac0_1 = jac0_1/length(files);
    
        jaccards(cnt).m = method;
        jaccards(cnt).prog = prog;
        jaccards(cnt).jac3 = jac3;
        jaccards(cnt).jac2 = jac2;
        jaccards(cnt).jac1 = jac1;
        jaccards(cnt).jac0 = jac0;
        jaccards(cnt).jac_1 = jac_1;
        cnt = cnt +1;
%         jaccards(t).jac21 = jac21;
%         jaccards(t).jac0_1 = jac0_1;
        
    end
end