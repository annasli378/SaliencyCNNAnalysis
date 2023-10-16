clear all,close all
%% paths
path_to_hm = "D:/Testy_metod/nauczone_modele_kfold/resnet/HM_resnet_TF2_k2_acc96/" ;
method = "AblationCAM/";

path_to_hm_method_dir = strcat(path_to_hm, "AblationCAM/" , 'HMwithIMG_title/');
path_to_hm_method = strcat(path_to_hm , method , 'pos/');
path_to_img = "D:/dataset_for_kfold5/k2/test/";

%%
files = dir(path_to_hm_method_dir);
files = files(3:end) ;

%%
for k=1:length(files)
     try
    m = imread(strcat(path_to_hm_method, files(k).name));
    m_norm = double(m) ./ 255;
    % znajdx obraz slm dla maski
    slm = imread(strcat(path_to_slm, files(k).name));

    [row, kol , dim] = size(slm);
    if dim>1
        slm = slm(:,:,1);
    end
    
    slm_score_1 = 0;
    slm_score_2 = 0;

    % obliczanie pierwszego wskaznika przez mnozenie z najwazniejszym tylko
    slm3 = double(slm==255);
    slm3_N = sum(sum(slm3));
    slm3xHM = slm3 .* m_norm;
    sum_slm3xHM = sum(sum(slm3xHM));
    slm_score_1 = sum_slm3xHM / slm3_N;

    % obliczanie drugiego wskaxnika przez mnozenie 3, 2, 1
    slm2 = double(slm==128);
    slm2_N = sum(sum(slm2)); 
    slm1 = double(slm==64);
    slm1_N = sum(sum(slm1)); 
    slm123 = slm3+slm2/2+slm1/4;
    slm123_N =  slm3_N+slm2_N+slm1_N;
    slm123xHM = slm123 .* m_norm;
    sum_slm123xHM = sum(sum(slm123xHM));
    slm_score_2 = sum_slm123xHM / slm123_N;

    SLM_SCORE(k).img_name = files(k).name;
    SLM_SCORE(k).slmscore1 = slm_score_1;
    SLM_SCORE(k).slmscore2 = slm_score_2;
     catch 
         disp(files(k).name)
     end

    
end