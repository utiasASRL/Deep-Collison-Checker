# Deep-Collison-Checker

In this repository, we share the implementation of the paper [Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes](https://arxiv.org/abs/2108.10585). This code is a minimalist version made with two objectives:
 - Easily reproduce the results presented in the paper
 - Possiblity to apply the network to other datasets

 As shown in the next figure, in this repo, we provide the code for our **automated annotation** and **network training**. The whole simulation is ignored, and we provide preprocessed data instead. 

![Intro figure](./Data/github_im.png)


## Setup

For convenience we provide a Dockerfile which builds a docker image able to run the code. Please refer to [Docker/README.md](./Docker/README.md) for detailed setup instructions


## Experiments

We provide scripts for three experiments: ModelNet40, S3DIS and SemanticKitti. The instructions to run these 
experiments are in the [doc](./doc) folder.

* [Object Classification](./doc/object_classification_guide.md): Instructions to train KP-CNN on an object classification
 task (Modelnet40).
 
* [Scene Segmentation](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on a scene segmentation 
 task (S3DIS).
 
* [SLAM Segmentation](./doc/slam_segmentation_guide.md): Instructions to train KP-FCNN on a slam segmentation 
 task (SemanticKitti).
 
* [Pretrained models](./doc/pretrained_models_guide.md): We provide pretrained weights and instructions to load them.
 
* [Visualization scripts](./doc/visualization_guide.md): For now only one visualization script has been implemented: 
the kernel deformations display.





## Introduction

Hugues Thomas, Matthieu Gallet de Saint Aurin, Jian Zhang, Timothy D. Barfoot


Have a scheme of the original file system and the corresponding docker volumes

Have an instalation guide

 - Set up the docker images
 - Compile the cpp wrappers for KPConv
 - Set up the catkin workspaces

Have some tutorial guides

 - Guide to run the training script alone with preprocessed data that we provide
 - Guide to run the simulator alone and generate data
 - Guide to run the training script on the generated data
 - Guide to show the results of the training
 - Guide to run simulator with a trained model
 - Guide for the development with visual studio: script to forward the tcp ports + visual studio attach + open workspace + Install extension in remote contatiner

## Reference

If you are to use this code, please cite our paper

```
@article{thomas2021learning,
    Author = {Thomas, Hugues and Gallet de Saint Aurin, Matthieu and Zhang, Jian and Barfoot, Timothy D.},
    Title = {Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes},
    Journal = {arXiv preprint arXiv:2108.10585},
    Year = {2021}
}
```

## License
Our code is released under MIT License (see LICENSE file for details).