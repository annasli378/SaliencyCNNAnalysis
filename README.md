# Saliency CNN Analysis
Attention analysis of convolutional neural networks for classification of skin ultrasound images

## About
When verifying the performance of neural networks, the issues of interpretability (interpreting), explainability (explaining) or understanding of artificial intelligence systems are raised. The latter is most often defined as the ability to characterize the model's behavior, its functioning, without penetrating the internal structure.

**Interpretability** - how easy it is for a person to understand why the model made a particular decision, i.e. assigned a specific label to the data being classified. In the case of images or texts, this concept can be defined as the mapping of the model's prediction (i.e., an abstract label, concept) to a known set of input data - pixels or separate words. 

**Explanation**  - a set of features from the interpreted domain, on the basis of which a decision was made. A graphical representation of such an explanation can be, for example, a heat map (HM), also known as an attention map. It provides a visual representation of which regions of the image have the greatest influence on a given classification, i.e. the assignment of a selected label by the model

## Dataset

Dataset source:

https://data.mendeley.com/datasets/5p7fxjt7vs/1

The analyzed data was recorded with a DUB SkinSkanner 75 camera and originally had dimensions of 2067 by 1555 pixels. The images were from both the areas lesions, as well as from healthy forearm skin in the case of the control group. Based on the segmented areas of epidermis, tumor lesions and SLEB layer were then used to obtain skin layer maps, used in the subsequent analysis of the focus areas of models taught to recognize selected classes on HFUS images.

## Methods

### Models

Four architectures, often used in medical image analysis issues, were selected for the study. The selected models are:
- DenseNet201
- GoogLeNet
- MobileNet v2 
- ResNet101

A k-fold cross-validation (k=5) was applied during the study. All images had a target size of 224 × 224 pixels. 

Transfer learning (ImageNet dataset) and also data augmentation (rotation and reflection in the vertical axis) were applied.

### Saliency methods


### Saliency based evaluation
The selected evaluation criteria, determined by the performance of the model on the test set, take into account:
- correct localization of areas by the model, as determined by the Hit Rate parameter,
- proper identification of details in the image, determined by the coverage of the heat map and skin layer map,
- the performance of the model on the test set.

Localization ability was assessed by the Hit Rate, which determines the hits of the peak of the relevance map to the relevant regions of the skin layer map. The calculation method took into account gradual changes in the importance of regions and scored hits on a scale of 3 to -1. It also partially rewarded in this way the focus of the model on regions below the most important area.

The resolution of the model was determined by the degree to which it focuses on the most important elements of the image. It was determined in two ways, based directly on the heat map and on the masks generated from it. 

In addition, the reliability of the method itself was also analyzed, that is, determining how much the generated attention map can be trusted. This was described by two coefficients - Drop in Confidence (DiC) and Increase in Confidence (IiC). 

Two methods were used to cover pixels in the original images:
- darkening by applying a heat map to the image,
- pixel zeroing by applying a binary mask obtained from thresholding the map

A decrease in the confidence of the classification of the changed area indicates that most likely the heat map or mask has covered most of the area relevant for classification. If confidence has not dropped, the HM or mask does not provide full information about the area on which the model is focused on. There is also the possibility that after removing the area in question, the remaining portion is enough to make a good classification, for example, if the mask is too small. If classification confidence does not increase after removing insignificant areas, it may be due to the method covering too much area or it may be due to the fact that the attention map does not give full information about the focus areas.

## Results

In order to evaluate the learned models, their performance was tested on a test set of 126 samples:
| model | accuracy [%] | precision [%]  | recall [%]  | f1-score [%]  |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DenseNet201  | 96.03  | 96.02  | 96.03  |  95.90  |
| GoogLeNet | 95.24  | 95.65  | 95.24  | 94.87 |
| MobileNet v2  | 94.44  | 94.99  | 94.44  | 93.92 |
| ResNet101 | 95.28  | 95.25  | 95.28  | 95.10  |

Summary of localization results for selected CAM methods:
RYS


Summary of localization results for selected methods: positive attributions (darker color) and negative attributions (lighter color):
RYS

### Examples


### Summary

## Bibliography

https://arxiv.org/abs/1312.6034  <br>
`Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
Scott M. Lundberg, Su-In Lee`

https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html  <br>
`A Unified Approach to Interpreting Model Predictions
Karen Simonyan, Andrea Vedaldi, Andrew Zisserman`

https://github.com/jacobgil/pytorch-grad-cam  <br>
`Advanced AI explainability for PyTorch
Jacob Gildenblat`

https://captum.ai/docs/attribution_algorithms  <br>
`Deep inside convolutional networks: Visualising image classification models and saliency maps
K. Simonyan, A. Vedaldi, A. Zisserman`

https://arxiv.org/abs/1312.6034  <br>
`Deep inside convolutional networks: Visualising image classification models and saliency maps
K. Simonyan, A. Vedaldi, A. Zisserman`

https://www.mdpi.com/1424-8220/21/17/5846 <br>
`Deep learning-based high-frequency ultrasound skin image classification with multicriteria model evaluation
J. Czajkowska, P. Badura, S. Korzekwa, A. Płatkowska-Szczerek, M. Słowińska`

https://pubmed.ncbi.nlm.nih.gov/35214381 <br>
` High-frequency ultrasound dataset for deep learning-based image quality assessment
J. Czajkowska, J. Juszczyk, L. Piejko, M. Glenc-Ambroży`

https://arxiv.org/abs/1610.02391 <br>
`Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra`

https://arxiv.org/abs/2011.08891 <br>
`Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks
Rachel L. Draelos, Lawrence Carin`

https://arxiv.org/abs/1710.11063 <br>
`Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian`

https://arxiv.org/abs/1910.01279 <br>
`Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, Xia Hu`

https://ieeexplore.ieee.org/abstract/document/9093360/ <br>
`Ablation-cam: Visual explanations for deep convolutional network via gradient-free localization.
Saurabh Desai and Harish G Ramaswamy. In WACV, pages 972–980, 2020`

https://arxiv.org/abs/2008.00299 <br>
`Eigen-CAM: Class Activation Map using Principal Components
Mohammed Bany Muhammad, Mohammed Yeasin`

http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf <br>
`LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
Peng-Tao Jiang; Chang-Bin Zhang; Qibin Hou; Ming-Ming Cheng; Yunchao Wei`

https://arxiv.org/abs/1905.00780 <br>
`Full-Gradient Representation for Neural Network Visualization
Suraj Srinivas, Francois Fleuret`

https://arxiv.org/pdf/1704.02685.pdf <br>
`Learning important features through propagating activation differences
A. Shrikumar, P. Greenside, A. Kundaje`

https://christophm.github.io/interpretable-ml-book <br>
`Interpretable Machine Learning
C. Molnar`



