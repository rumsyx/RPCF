RPCF
==
Code for the paper 'ROI Pooled Correlation Filters for Visual Tracking' (CVPR 2019)

Paper Link
---
* [Paper](https://arxiv.org/pdf/1911.01668.pdf)


Installation
---

1. Clone the GIT repository

2. Compile the source code in the ./caffe directory and the matlab interface following the installation instruction of caffe.

3. Download the VGG_ILSVRC_16_layers.caffemodel (553.4 MB) from https://gist.github.com/ksimonyan/211839e770f7b538e2d8, and put the caffemodel file under the ./model directory.

4. Download imagenet-vgg-m-2048 (345 MB) from http://www.vlfeat.org/matconvnet/pretrained/, and put the file into ./networks . 

5. Compile matconvnet in the ./external_libs folders.

6. Run the demo code demo_RPCF.m to test the code. You can customize your own test sequences following this example.

7. Modify the configSeq.m to your OTB dataset path, then run run_RPCF.m on all 100 datasets.


Results
---
* [RESULTS](https://pan.baidu.com/s/1eUDp4lXmtdo9awIVGE6Vaw)  (Extracted code: 2cdc )
 
 The above link includs the results of OTB-100、VOT-2018 datasets.


Citation
---
Please cite:
```
@inproceedings{sun2019roi,
  title={ROI Pooled Correlation Filters for Visual Tracking},
  author={Sun, Yuxuan and Sun, Chong and Wang, Dong and He, You and Lu, Huchuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5783--5791},
  year={2019}
}
```


Environment
---
Ubuntu 14.04

MATLAB R2017a

Nvidia 1080 GPU

CUDA 8.0

CUDNN 6.0

