### FSS_MPSNet

### Datasets and pre-processing
Download:  
1. **Abdominal MRI**  [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)  
2. **Abdominal CT**   [the MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)
3. **Cardiac MRI** [Multi-sequence Cardiac MRI Segmentation dataset (bSSFP fold)](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/)  

# Data processing
1. datasets_chaost2.py
2. datasets_cmr.py
3. datasets_sabs.py

**Pre-processing** is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and we follow the procedure on their github repository.  

**Supervoxel segmentation** is performed according to [Hansen et al.](https://github.com/sha168/ADNet.git) and we follow the procedure on their github repository.  

### Training  
1. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth).  
2. Run train.py

### Testing
Run test.py

### Acknowledgement
This code is based on [SSL-ALPNet](https://arxiv.org/abs/2007.09886v2) (ECCV'20) by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and [ADNet](https://www.sciencedirect.com/science/article/pii/S1361841522000378) by [Hansen et al.](https://github.com/sha168/ADNet.git). 
