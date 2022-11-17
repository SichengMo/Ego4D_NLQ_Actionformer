# A Simple Transformer-Based Model for Ego4D Natural Language Queries Challenge

The second place of Ego4D Natural Language Queries challenge on ECCV 2022. Our arXiv version can be found in [this link](https://arxiv.org/abs/2211.08704). We invite our audience to try out the code.

## Introduction

This code repo implements an [ActionFormer](https://github.com/happyharrycn/actionformer_release) variant for single-stage temporal sentence grounding on Ego4D NLQ challenge. Our model differs from ActionFormer in following aspects:

* An additional transformer-based text encoder.
* Transformer-based classification and regression heads.
* Attention-based fusion of video and text features.
* Frame level contrastive loss


## Code Overview
The structure of this code repo is heavily inspired by [ActionFormer](https://github.com/happyharrycn/actionformer_release). Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/model: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

## Dataset
### Ego4D NLQ
* Download *ego_vlp_reshape.zip* from [this google drive link](https://drive.google.com/file/d/1MnurvQQsBRdm3fn7RT6vLLWCfFkzL7bL/view?usp=share_link). This file includes EgoVLP feature in pt format.
* Download the official Slowfast and Omnivore features from [Ego4D official repo](https://github.com/facebookresearch/Ego4d/tree/main/ego4d/cli). 

**Details** These are EgoVlP features extracted using [EgoVLP official code](https://github.com/showlab/EgoVLP). The features are extracted using clips of `16 frames` and a stride of `16 frames`. We reshaped these features to alien with Ego4D official slowfast features.

## Quick Start
* Follow data/DATA_README.md for prepare video features.
* Unpack the file under *./data/ego4d* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───ego4d/
│    │	 └───annotations
│    │	 └───video_features
│    │	     └───ego_vlp_reshape
│    │	     └───official_slowfast   
│    │	     └───official_omnivore 
│    │	     └───fusion     
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our model on the Ego4D dataset. This will create an experiment folder under *./log* that stores training config, logs, and checkpoints.
```shell
python ./train.py --config configs/ego4d.yaml -n ego4d -g 0
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./log/ego4d/
```
* Evaluate the trained model. The Rank1@IoU0.3 metric for Ego4D should be around 15.5%.
```shell
python ./eval.py -n ego4d -c last -ema 
```

* Generate submission file for Ego4D NLQ challenge.
```shell
python ./submit.py -n ego4d -c last -ema 
```

## Contact
Sicheng Mo (smo3@wisc.edu)

## References
If you are using our code, please consider citing our paper.
```
@misc{mo2022simple,
      title={A Simple Transformer-Based Model for Ego4D Natural Language Queries Challenge}, 
      author={Sicheng Mo and Fangzhou Mu and Yin Li},
      year={2022},
      eprint={2211.08704},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```