# VOS

This is the source code accompanying the paper [***VOS: Learning What You Don’t Know by Virtual Outlier Synthesis***](https://openreview.net/forum?id=TW7d65uYu5M) by Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li


The codebase is heavily based on [ProbDet](https://github.com/asharakeh/probdet) and [Detectron2](https://github.com/facebookresearch/detectron2).

## Ads 

Checkout our CVPR'22 work [STUD](https://github.com/deeplearning-wisc/stud) on object detection in video datasets,  NeurIPS'22 work [SIREN](https://github.com/deeplearning-wisc/siren) on OOD detection for detection transformers if you are interested!

## Update

02/05/2023---we have uploaded the code [here](https://github.com/deeplearning-wisc/vos/blob/main/plot_fig1) for reproducing the figure 1 of our paper.

05/08/2022---We have updated the openreview with the new results of using a nonlinear MLP for binary classification instead of the logistic regression, which is slightly better. Please check the code and models [here](https://github.com/deeplearning-wisc/vos/tree/main-MLP).

## Requirements
```
pip install -r requirements.txt
```
In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Dataset Preparation

**PASCAL VOC**

Download the processed VOC 2007 and 2012 dataset from [here](https://drive.google.com/file/d/1n9C4CiBURMSCZy2LStBQTzR17rD_a67e/view?usp=sharing).

The VOC dataset folder should have the following structure:
<br>

     └── VOC_DATASET_ROOT
         |
         ├── JPEGImages
         ├── voc0712_train_all.json
         └── val_coco_format.json

**COCO**

Download COCO2017 dataset from the [official website](https://cocodataset.org/#home). 

Download the OOD dataset (json file) when the in-distribution dataset is Pascal VOC from [here](https://drive.google.com/file/d/1Wsg9yBcrTt2UlgBcf7lMKCw19fPXpESF/view?usp=sharing). 

Download the OOD dataset (json file) when the in-distribution dataset is BDD-100k from [here](https://drive.google.com/file/d/1AOYAJC5Z5NzrLl5IIJbZD4bbrZpo0XPh/view?usp=sharing).

Put the two processed OOD json files to ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── COCO_DATASET_ROOT
         |
         ├── annotations
            ├── xxx (the original json files)
            ├── instances_val2017_ood_wrt_bdd_rm_overlap.json
            └── instances_val2017_ood_rm_overlap.json
         ├── train2017
         └── val2017

**BDD-100k**

Donwload the BDD-100k images from the [official website](https://bdd-data.berkeley.edu/).

Download the processed BDD-100k json files from [here](https://drive.google.com/file/d/1ZbbdKEakSjyOci7Ggm046hCCGYqIHcbE/view?usp=sharing) and [here](https://drive.google.com/file/d/1Rxb9-6BUUGZ_VsNZy9S2pWM8Q5goxrXY/view?usp=sharing).

The BDD dataset folder should have the following structure:
<br>

     └── BDD_DATASET_ROOT
         |
         ├── images
         ├── val_bdd_converted.json
         └── train_bdd_converted.json
**OpenImages**

Download our OpenImages validation splits [here](https://drive.google.com/file/d/1UPuxoE1ZqCfCZX48H7bWX7GGIJsTUrt5/view?usp=sharing). We created a tarball that contains the out-of-distribution data splits used in our paper for hyperparameter tuning. Do not modify or rename the internal folders as those paths are hard coded in the dataset reader. The OpenImages dataset is created in a similar way following this [paper](https://openreview.net/forum?id=YLewtnvKgR7&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2021%2FConference%2FAuthors%23your-submissions)). 

The OpenImages dataset folder should have the following structure:
<br>

     └── OEPNIMAGES_DATASET_ROOT
         |
         ├── coco_classes
         └── ood_classes_rm_overlap



**Visualization of the OOD datasets**

 The OOD images with respect to different in-distribution datasets can be downloaded from [ID-VOC-OOD-COCO](https://drive.google.com/drive/folders/1NxodhoxTX5YBHJWHAa6tB2Ta1oxoTfzu?usp=sharing), [ID-VOC-OOD-openimages](https://drive.google.com/drive/folders/1pRP7CAWG7naDECfejo03cl7PF3VJEjrn?usp=sharing), [ID-BDD-OOD-COCO](https://drive.google.com/drive/folders/1Wgmfcp2Gd3YvYVyoRRBWUiwwKYXJeuo8?usp=sharing), [ID-BDD-OOD-openimages](https://drive.google.com/drive/folders/1LyOFqSm2G8x7d2xUkXma2pFJVOLgm3IQ?usp=sharing).


## Training

Firstly, enter the detection folder by running
```
cd detection
```

Before training, modify the dataset address by changing "dataset-dir" according to your local dataset address.

**Vanilla Faster-RCNN with VOC as the in-distribution dataset**
```

python train_net.py
--dataset-dir path/to/dataset/dir
--num-gpus 8
--config-file VOC-Detection/faster-rcnn/vanilla.yaml 
--random-seed 0 
--resume
```
**Vanilla Faster-RCNN with BDD as the in-distribution dataset**
```
python train_net.py 
--dataset-dir path/to/dataset/dir
--num-gpus 8 
--config-file BDD-Detection/faster-rcnn/vanilla.yaml 
--random-seed 0 
--resume
```
**VOS on ResNet**
```
python train_net_gmm.py 
--dataset-dir path/to/dataset/dir
--num-gpus 8 
--config-file VOC-Detection/faster-rcnn/vos.yaml 
--random-seed 0 
--resume
```
**VOS on RegNet**

Before training using the RegNet as the backbone, download the pretrained RegNet backbone from [here](https://drive.google.com/file/d/1WyE_OIpzV_0E_Y3KF4UVxIZJTSqB7cPO/view?usp=sharing).
```
python train_net_gmm.py 
--dataset-dir path/to/dataset/dir
--num-gpus 8 
--config-file VOC-Detection/faster-rcnn/regnetx.yaml 
--random-seed 0 
--resume
```
Before training on VOS, change "VOS.STARTING_ITER" and "VOS.SAMPLE_NUMBER" in the config file to the desired numbers in paper.

## Evaluation

**Evaluation with the in-distribution dataset to be VOC**

Firstly run on the in-distribution dataset:
```
python apply_net.py 
--dataset-dir path/to/dataset/dir
--test-dataset voc_custom_val 
--config-file VOC-Detection/faster-rcnn/vos.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Then run on the OOD dataset:

```
python apply_net.py
--dataset-dir path/to/dataset/dir
--test-dataset coco_ood_val 
--config-file VOC-Detection/faster-rcnn/vos.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Obtain the metrics using:
```
python voc_coco_plot.py 
--name vos 
--thres xxx 
--energy 1 
--seed 0
```
Here the threshold is determined according to [ProbDet](https://github.com/asharakeh/probdet). It will be displayed in the screen as you finish evaluating on the in-distribution dataset.

**Evaluation with the in-distribution dataset to be BDD**

Firstly run on the in-distribution dataset:
```
python apply_net.py 
--dataset-dir path/to/dataset/dir
--test-dataset bdd_custom_val 
--config-file BDD-Detection/faster-rcnn/vos.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Then run on the OOD dataset:

```
python apply_net.py 
--dataset-dir path/to/dataset/dir
--test-dataset coco_ood_val_bdd 
--config-file BDD-Detection/faster-rcnn/vos.yaml 
--inference-config Inference/standard_nms.yaml 
--random-seed 0 
--image-corruption-level 0 
--visualize 0
```
Obtain the metrics using:
```
python bdd_coco_plot.py
--name vos 
--thres xxx 
--energy 1 
--seed 0
```
**Pretrained models**

The pretrained models for Pascal-VOC can be downloaded from [vanilla](https://drive.google.com/file/d/10tfHHSdjXPds8ol46HU3G62zPS5tCcTu/view?usp=sharing) and [VOS-ResNet](https://drive.google.com/file/d/1Q8Gq9QhacBDwfIPHdE6ArNaAqaJQHJzl/view?usp=sharing) and [VOS-RegNet](https://drive.google.com/file/d/1W4yltGF-wvzwEuImaDwMLT-GeOnhcdaB/view?usp=sharing).

The pretrained models for BDD-100k can be downloaded from [vanilla](https://drive.google.com/file/d/16D4GOlSPSrY9-Y0gZlB13NczZAegjF1D/view?usp=sharing) and [VOS-ResNet](https://drive.google.com/file/d/1CJoweDQkNUDi2Gz8qeA1DruaBQig6i1b/view?usp=sharing) and [VOS-RegNet](https://drive.google.com/file/d/1DFBkYLSTLJaQpr3mC0BGLQf04h5vRFhp/view?usp=sharing).


## VOS on Classification models

**Train on WideResNet**
```
cd classification/CIFAR/ & 
python train_virtual.py 
--start_epoch 40 
--sample_number 1000 
--sample_from 10000 
--select 1 
--loss_weight 0.1 
```
where "start_epoch" denotes the starting epoch of the uncertainty regularization branch.

"sample_number" denotes the size of the in-distribution queue.

"sample_from" and "select" are used to approximate the likelihood threshold during virtual outlier synthesis.

"loss_weight" denotes the weight of the regularization loss.

Please see Section 3 and Section 4.1 in the paper for details. 

**Train on DenseNet**
```
cd classification/CIFAR/ &
python train_virtual_dense.py 
--start_epoch 40 
--sample_number 1000 
--sample_from 10000 
--select 1 
--loss_weight 0.1 
```

**Evaluation on different classifiers**

```
cd classification/CIFAR/ & 
python test.py 
--model_name xx 
--method_name xx 
--score energy 
--num_to_avg 10
```

where "model_name" denotes the model architectures. ("res" denotes the WideResNet and "dense" denotes the DenseNet.)

"method_name" denotes the checkpoint name you are loading. 

**Pretrained models**

We provide the pretrained models using [WideResNet](https://drive.google.com/file/d/19fIqKrvHBajEpUWWwDMI8c62tYFOvO5n/view?usp=sharing) and  [DenseNet](https://drive.google.com/file/d/1QTHFGFMIcdWVmqMMfzlwu4q-xgkQELq8/view?usp=sharing) with the in-distribution dataset to be CIFAR-10.


## Citation ##
If you found any part of this code is useful in your research, please consider citing our paper:

```
 @article{du2022vos,
      title={VOS: Learning What You Don’t Know by Virtual Outlier Synthesis}, 
      author={Du, Xuefeng and Wang, Zhaoning and Cai, Mu and Li, Yixuan},
      journal={Proceedings of the International Conference on Learning Representations},
      year={2022}
}
```

